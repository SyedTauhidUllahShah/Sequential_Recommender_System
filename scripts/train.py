import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sasrec_model import SASRec

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

def train_sasrec(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_seqs, target_items in train_loader:
        input_seqs, target_items = input_seqs.to(device), target_items.to(device)
        optimizer.zero_grad()
        output = model(input_seqs)
        output = output.view(-1, output.size(-1))  # Flatten to [batch_size * seq_len, vocab_size]
        target_items = target_items.view(-1)
        loss = criterion(output, target_items)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_sasrec(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_seqs, target_items in val_loader:
            input_seqs, target_items = input_seqs.to(device), target_items.to(device)
            output = model(input_seqs)
            output = output.view(-1, output.size(-1))  # Flatten to [batch_size * seq_len, vocab_size]
            target_items = target_items.view(-1)
            loss = criterion(output, target_items)
            total_loss += loss.item()
    return total_loss / len(val_loader)
