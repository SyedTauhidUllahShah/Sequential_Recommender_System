import torch
import numpy as np
from sasrec_model import SASRec
from train import SequenceDataset
from torch.utils.data import DataLoader

def evaluate_metrics(predicted_items, actual_items, ks=[5, 10, 20]):
    """
    Evaluate Precision@K, Recall@K, MRR, HR@10, NDCG@10 for different values of K.
    """
    actual_items = set(actual_items)
    metrics = {}
    
    for k in ks:
        predicted_at_k = set(predicted_items[:k])
        true_positives = len(predicted_at_k & actual_items)

        precision_at_k = true_positives / k
        recall_at_k = true_positives / len(actual_items)

        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for rank, item in enumerate(predicted_items):
            if item in actual_items:
                mrr = 1 / (rank + 1)
                break

        # HR@K (Hit Rate)
        hit_rate = 1 if len(predicted_at_k & actual_items) > 0 else 0

        # NDCG@K
        dcg = 0
        idcg = 1
        for rank, item in enumerate(predicted_items[:k]):
            if item in actual_items:
                dcg += 1 / np.log2(rank + 2)
        ndcg = dcg / idcg

        metrics[f'Precision@{k}'] = precision_at_k
        metrics[f'Recall@{k}'] = recall_at_k
        metrics[f'MRR@{k}'] = mrr
        metrics[f'HR@{k}'] = hit_rate
        metrics[f'NDCG@{k}'] = ndcg

    return metrics

def evaluate_sasrec(model, test_loader, ks=[5, 10, 20]):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for input_seqs, target_items in test_loader:
            predicted_items = model(input_seqs).argmax(dim=-1)
            for predicted, actual in zip(predicted_items, target_items):
                metrics = evaluate_metrics(predicted.tolist(), actual.tolist(), ks)
                all_metrics.append(metrics)
    return all_metrics
