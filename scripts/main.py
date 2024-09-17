import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sasrec_model import SASRec
from train import train_sasrec, validate_sasrec, SequenceDataset

def main(args):
    # Load data from 'data/processed'
    train_seqs = np.load('data/processed/train_sequences.npy', allow_pickle=True)
    val_seqs = np.load('data/processed/val_sequences.npy', allow_pickle=True)
    train_targets = np.load('data/processed/train_targets.npy', allow_pickle=True)
    val_targets = np.load('data/processed/val_targets.npy', allow_pickle=True)

    train_dataset = SequenceDataset(train_seqs, train_targets)
    val_dataset = SequenceDataset(val_seqs, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, optimizer, and loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRec(args.num_items, args.hidden_units, args.num_blocks, args.num_heads, args.max_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        train_loss = train_sasrec(model, train_loader, optimizer, criterion, device)
        val_loss = validate_sasrec(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_sasrec_model.pth')  # Save best model to 'models'

    # Save losses for visualization
    np.save('scripts/train_losses.npy', train_losses)
    np.save('scripts/val_losses.npy', val_losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_items', type=int, default=1000)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    main(args)
