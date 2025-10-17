import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear layers to transform input into Query, Key, and Value
        self.Q_proj = nn.Linear(embed_dim, embed_dim)
        self.K_proj= nn.Linear(embed_dim, embed_dim)
        self.V_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)

        # Project input to Query, Key, Value
        Q = self.Q_proj(x)  # (batch_size, sequence_length, embed_dim)
        K = self.K_proj(x)      # (batch_size, sequence_length, embed_dim)
        V = self.V_proj(x)  # (batch_size, sequence_length, embed_dim)

        # Calculate attention scores (dot product of queries and keys)
        # Transpose keys for matrix multiplication: (batch_size, embed_dim, sequence_length)
        scores = torch.matmul(Q, K.transpose(-2, -1)) 
        # scores shape: (batch_size, sequence_length, sequence_length)

        # Scale scores to prevent vanishing gradients
        scaling_factor = self.embed_dim**0.5
        scaled_scores = scores / scaling_factor

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scaled_scores, dim=-1)
        # attention_weights shape: (batch_size, sequence_length, sequence_length)

        # Multiply attention weights by values to get context vectors
        context_vectors = torch.matmul(attention_weights, V)
        # context_vectors shape: (batch_size, sequence_length, embed_dim)

        return context_vectors