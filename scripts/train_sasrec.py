
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

if __name__ == '__main__':
    num_items = 1000
    hidden_units = 128
    num_blocks = 2
    num_heads = 2
    max_seq_len = 50
    batch_size = 32
    num_epochs = 5
    lr = 0.001

    # Load data
    train_seqs = np.load('data/processed/train_sequences.npy', allow_pickle=True)
    val_seqs = np.load('data/processed/val_sequences.npy', allow_pickle=True)
    train_targets = np.load('data/processed/train_targets.npy', allow_pickle=True)
    val_targets = np.load('data/processed/val_targets.npy', allow_pickle=True)
    
    train_dataset = SequenceDataset(train_seqs, train_targets)
    val_dataset = SequenceDataset(val_seqs, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(num_items, hidden_units, num_blocks, num_heads, max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_sasrec(model, train_loader, optimizer, criterion, device)
        val_loss = validate_sasrec(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_sasrec_model.pth')  # Save the best model
