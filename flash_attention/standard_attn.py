"""
Standard (non-flash) attention, written out by hand as a baseline.

Shape notation used throughout:
    B = batch size, N = sequence length, C = embed_dim,
    H = num_heads,  d = head_dim = C // H
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head self-attention transformer block.

    Args:
        embed_dim: size of each token embedding (C).

    Input/output shape: (B, N, C).
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        # Output projection (W_O), applied to the attention output before the residual add
        self.out = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: (B, N, C) -> (B, N, C)"""
        Q = self.q(x)  # (B, N, C)
        K = self.k(x)  # (B, N, C)
        V = self.v(x)  # (B, N, C)

        B, N, C = x.shape

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (B, N, N)
        attn_probs = torch.softmax(attn_scores, dim=-1)                   # (B, N, N)
        attn_output = torch.matmul(attn_probs, V)                         # (B, N, C)
        attn_output = self.out(attn_output)                               # (B, N, C)

        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        return x


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

    def __init__(self, embed_dim, num_heads):
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

    def forward(self, x):
        """x: (B, N, C) -> (B, N, C)"""
        B, N, C = x.shape

        Q = self.q(x)  # (B, N, C)
        K = self.k(x)  # (B, N, C)
        V = self.v(x)  # (B, N, C)

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        attn_probs = torch.softmax(attn_scores, dim=-1)                               # (B, H, N, N)
        attn_output = torch.matmul(attn_probs, V)                                     # (B, H, N, d)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.out(attn_output)                                           # (B, N, C)

        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        return x


# ----------------------------------------------------------------------------
# TEST CASES BELOW
# ----------------------------------------------------------------------------

def reference_forward(m, x, num_heads):
    """
    Run block `m` with PyTorch's built-in attention in place of the hand-written math.

    Uses m's own weights (q, k, v, out, norms, mlp) and swaps only the attention
    core for F.scaled_dot_product_attention. If m(x) matches this output, the
    hand-written attention math in m.forward is correct.

    Args:
        m: a SelfAttention or MultiHeadSelfAttention block.
        x: input of shape (B, N, C).
        num_heads: number of heads to split into (1 for SelfAttention).

    Returns:
        Tensor of shape (B, N, C).
    """
    B, N, C = x.shape
    head_dim = C // num_heads

    def split_heads(t):  # (B, N, C) -> (B, num_heads, N, head_dim)
        return t.view(B, N, num_heads, head_dim).transpose(1, 2)

    q, k, v = split_heads(m.q(x)), split_heads(m.k(x)), split_heads(m.v(x))
    attn = F.scaled_dot_product_attention(q, k, v)  # (B, num_heads, N, head_dim)
    attn = attn.transpose(1, 2).reshape(B, N, C)    # merge heads
    attn = m.out(attn)

    x = m.norm1(x + attn)
    x = m.norm2(x + m.mlp(x))
    return x


def test_self_attention():
    """Check SelfAttention against reference_forward with 1 head, over several shapes."""
    test_inputs = [  # (B, N, C)
        (1, 1, 8),
        (2, 5, 16),
        (3, 17, 64)
    ]
    for B, N, C in test_inputs:
        torch.manual_seed(0)
        m = SelfAttention(C).double()
        x = torch.randn(B, N, C, dtype=torch.float64)

        y_ref = reference_forward(m, x, 1)
        y_test = m(x)
        torch.testing.assert_close(y_test, y_ref, msg=lambda err, shape=(B, N, C): f"Mismatch for input shape {shape}\n{err}")
    print("All tests passed.")


def test_multihead_self_attention():
    """Check MultiHeadSelfAttention against reference_forward over several shapes and head counts."""
    test_inputs = [  # (B, N, C, num_heads)
        (1, 1, 8, 1), # edge case: single token
        (2, 5, 16, 4),
        (3, 17, 64, 8), # odd sequence length
        (2, 128, 128, 16),
    ]
    for B, N, C, num_heads in test_inputs:
        torch.manual_seed(0)
        m = MultiHeadSelfAttention(C, num_heads=num_heads).double()
        x = torch.randn(B, N, C, dtype=torch.float64)

        y_ref = reference_forward(m, x, m.num_heads)
        y_test = m(x)
        torch.testing.assert_close(y_test, y_ref, msg=lambda err, shape=(B, N, C), h=num_heads: f"Mismatch for input shape {shape} with {h} heads\n{err}")
    print("All multihead tests passed.")


if __name__ == "__main__":
    test_self_attention()
    test_multihead_self_attention()
