
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, num_items, hidden_units, num_blocks, num_heads, max_seq_len):
        super(SASRec, self).__init__()
        self.item_embedding = nn.Embedding(num_items, hidden_units)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_units)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units, 
                nhead=num_heads,
                dim_feedforward=hidden_units * 4
            ) for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.fc = nn.Linear(hidden_units, num_items)

    def forward(self, input_seq):
        # Embedding the item ids
        seq_embeddings = self.item_embedding(input_seq)
        pos_ids = torch.arange(seq_embeddings.size(1), dtype=torch.long, device=input_seq.device)
        pos_embeddings = self.pos_embedding(pos_ids)
        embeddings = seq_embeddings + pos_embeddings

        # Passing through multiple transformer layers
        embeddings = self.layer_norm(embeddings)
        for block in self.blocks:
            embeddings = block(embeddings)

        # Predict next items
        return self.fc(embeddings)
