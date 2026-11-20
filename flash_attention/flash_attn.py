import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_attention(Q, K, V):
    """Standard attention core. Q, K, V: (B, H, N, d) -> (B, H, N, d)"""
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # (B, H, N, N)
    attn_probs = torch.softmax(attn_scores, dim=-1)                               # (B, H, N, N)
    attn_output = torch.matmul(attn_probs, V)                                     # (B, H, N, d)
    return attn_output


def flash_attention(Q, K, V, block_q=64, block_k=64):
    """
    Flash attention forward pass: tiled attention with an online softmax.
    
    Q: (..., N, d), K and V: (..., S, d) -> output (..., N, d)
    The leading dims (...) can be anything, e.g. (B, H) or nothing at all.
    """
    N = Q.shape[-2] # number of queries
    d = Q.shape[-1] # head dimension
    S = K.shape[-2] # number of keys (usually equal to N for self-attention)
    batch_shape = Q.shape[:-2] # all the leading dims, e.g. (B, H); empty for a plain (N, d)

    O = torch.zeros_like(Q) # (..., N, d), same dtype and device as Q

    for q_start in range(0, N, block_q):
        q_end = min(q_start + block_q, N)
        Q_blk = Q[..., q_start:q_end, :] # (..., bq, d)
        bq = q_end - q_start

        m = torch.full((*batch_shape, bq), float('-inf'), dtype=Q.dtype, device=Q.device) # (..., bq)
        l = torch.zeros((*batch_shape, bq), dtype=Q.dtype, device=Q.device) # (..., bq)
        acc = torch.zeros((*batch_shape, bq, d), dtype=Q.dtype, device=Q.device) # (..., bq, d)

        for k_start in range(0, S, block_k):
            k_end = min(k_start + block_k, S)
            K_blk = K[..., k_start:k_end, :] # (..., bk, d)
            V_blk = V[..., k_start:k_end, :] # (..., bk, d)

            s_blk = Q_blk @ K_blk.transpose(-2, -1) / math.sqrt(d) # (..., bq, bk)

            m_new = torch.maximum(m, s_blk.amax(dim=-1)) # (..., bq)
            alpha = torch.exp(m - m_new) # (..., bq)
            p = torch.exp(s_blk - m_new[..., None]) # (..., bq, bk)

            l = alpha * l + p.sum(dim=-1) # (..., bq)
            acc = alpha[..., None] * acc + (p @ V_blk) # (..., bq, d)
            m = m_new # (..., bq)

        O[..., q_start:q_end, :] = acc / l[..., None] # (..., bq, d)

    return O


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention transformer block.

    Each head runs attention independently, then the heads are concatenated and mixed by
    the output projection.

    Args:
        embed_dim: size of each token embedding (C). Must be divisible by num_heads.
        num_heads: number of attention heads (H).

    Input/output shape: (B, N, C).
    """

    def __init__(self, embed_dim, num_heads, use_flash=False, block_q=64, block_k=64):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.out = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.use_flash = use_flash
        self.block_q = block_q
        self.block_k = block_k

    def forward(self, x):
        """x: (B, N, C) -> (B, N, C)"""
        B, N, C = x.shape

        Q = self.q(x)  # (B, N, C)
        K = self.k(x)  # (B, N, C)
        V = self.v(x)  # (B, N, C)

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            attn_output = flash_attention(Q, K, V, block_q=self.block_q, block_k=self.block_k)  # (B, H, N, d)
        else:
            attn_output = standard_attention(Q, K, V)  # (B, H, N, d)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.out(attn_output)                                           # (B, N, C)

        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        return x


# ----------------------------------------------------------------------------
# TEST CASES BELOW
# ----------------------------------------------------------------------------

def test_flash_attention_matches_sdpa():
    test_inputs = [ # (Q shape, K/V shape, block_q, block_k)
        ((10, 4), (10, 4), 4, 3), # plain (N, d), no batch dims
        ((1, 1, 1, 8), (1, 1, 1, 8), 64, 64), # single token
        ((2, 4, 10, 4), (2, 4, 10, 4), 4, 3),
        ((2, 4, 197, 32), (2, 4, 197, 32), 64, 48), # N not a multiple of either block size
        ((2, 4, 200, 32), (2, 4, 333, 32), 64, 48), # N != S
    ]

    for q_shape, kv_shape, block_q, block_k in test_inputs:
        torch.manual_seed(0)
        Q = torch.randn(*q_shape, dtype=torch.float64)
        K = torch.randn(*kv_shape, dtype=torch.float64)
        V = torch.randn(*kv_shape, dtype=torch.float64)

        out_ref = F.scaled_dot_product_attention(Q, K, V)
        out_flash = flash_attention(Q, K, V, block_q=block_q, block_k=block_k)
        torch.testing.assert_close(out_flash, out_ref, msg=lambda err, shape=q_shape: f"Mismatch for Q shape {shape}\n{err}")
    print("flash_attention matches SDPA.")


def test_flash_block_matches_standard_block():
    test_inputs = [ # (B, N, C, num_heads)
        (1, 1, 8, 1),
        (2, 5, 16, 4),
        (3, 17, 64, 8),
        (2, 128, 128, 16),
    ]
    for B, N, C, num_heads in test_inputs:
        torch.manual_seed(0)
        std = MultiHeadSelfAttention(C, num_heads).double()
        flash = MultiHeadSelfAttention(C, num_heads, use_flash=True, block_q=16, block_k=12).double()
        flash.load_state_dict(std.state_dict()) # copy weights so the blocks are identical

        x = torch.randn(B, N, C, dtype=torch.float64)
        torch.testing.assert_close(flash(x), std(x), msg=lambda err, shape=x.shape: f"Mismatch for input shape {shape}\n{err}")
    print("Flash block matches standard block")


def test_flash_gradients_match():
    torch.manual_seed(0)
    std = MultiHeadSelfAttention(32, 4).double()
    flash = MultiHeadSelfAttention(32, 4, use_flash=True, block_q=8, block_k=6).double()
    flash.load_state_dict(std.state_dict()) # copy weights so the blocks are identical

    x1 = torch.randn(2, 20, 32, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    w = torch.randn(2, 20, 32, dtype=torch.float64) # random weights so the loss isn't constnat
    (std(x1) * w).sum().backward()
    (flash(x2) * w).sum().backward()

    torch.testing.assert_close(x2.grad, x1.grad, msg=lambda err: f"Gradient mismatch between standard and flash attention\n{err}")
    print("Flash gradients match standard gradients")


if __name__ == "__main__":
    test_flash_attention_matches_sdpa()
    test_flash_block_matches_standard_block()
    test_flash_gradients_match()
