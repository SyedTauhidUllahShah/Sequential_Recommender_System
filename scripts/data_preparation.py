import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Load JSONL data from the original dataset
def load_json_data(file_path):
    json_list = []
    with open(file_path, 'r') as file:
        for json_object in file:
            json_dict = json.loads(json_object.strip())
            json_list.append(json_dict)
    return pd.DataFrame(json_list)

# Prepare data for SASRec model by creating user-item sequences
def prepare_data(max_seq_len=50):
    PATH_REVIEWS_DATA = "data/raw/Appliances.jsonl"
    PATH_META_DATA = "data/raw/meta_Appliances.jsonl"

    reviews_df = load_json_data(PATH_REVIEWS_DATA)
    meta_df = load_json_data(PATH_META_DATA)

    # Merge reviews with product metadata
    full_df = reviews_df.merge(meta_df, on="parent_asin", how="inner")

    # Process sequences
    full_df["user_id"] = "<|user_" + full_df["user_id"] + "|>"
    full_df["item_id"] = "<|item_" + full_df["parent_asin"] + "|>"

    # Sort by user and timestamp
    full_df = full_df.sort_values(by=["user_id", "timestamp"])

    user_sequences = []
    target_items = []

    # Group by user and create sequences
    for user_id, group in full_df.groupby('user_id'):
        items = group['item_id'].tolist()

        # Generate sequences of items for each user
        for i in range(1, len(items)):
            seq = items[max(0, i - max_seq_len):i]  # Previous interactions
            user_sequences.append(seq)
            target_items.append(items[i])  # The next item to predict

    return user_sequences, target_items

def split_data(sequences, targets, val_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    """
    train_seqs, temp_seqs, train_targets, temp_targets = train_test_split(
        sequences, targets, test_size=val_size + test_size, random_state=42)
    val_seqs, test_seqs, val_targets, test_targets = train_test_split(
        temp_seqs, temp_targets, test_size=test_size / (val_size + test_size), random_state=42)
    
    return train_seqs, val_seqs, test_seqs, train_targets, val_targets, test_targets

if __name__ == '__main__':
    sequences, targets = prepare_data()

    # Split the data
    train_seqs, val_seqs, test_seqs, train_targets, val_targets, test_targets = split_data(sequences, targets)

    # Save the processed data
    np.save('data/processed/train_sequences.npy', train_seqs)
    np.save('data/processed/val_sequences.npy', val_seqs)
    np.save('data/processed/test_sequences.npy', test_seqs)
    np.save('data/processed/train_targets.npy', train_targets)
    np.save('data/processed/val_targets.npy', val_targets)
    np.save('data/processed/test_targets.npy', test_targets)
