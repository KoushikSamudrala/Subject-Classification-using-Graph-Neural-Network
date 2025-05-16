import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import KFold
from src.model import GCN
from tqdm import tqdm

def train(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        correct = preds[test_mask] == data.y[test_mask]
        acc = int(correct.sum()) / int(test_mask.sum())
        return acc, preds[test_mask].cpu().numpy()

def get_mask(indices, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask

def main():
    # Load data
    dataset = Planetoid(root='data/raw', name='Cora')
    data = dataset[0]
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    # Class names as per Cora
    class_names = [
        'Case_Based',
        'Genetic_Algorithms',
        'Neural_Networks',
        'Probabilistic_Methods',
        'Reinforcement_Learning',
        'Rule_Learning',
        'Theory'
    ]

    # Prepare for 10-fold CV
    indices = np.arange(num_nodes)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # This will store the out-of-fold predictions
    all_preds = np.empty(num_nodes, dtype=int)
    all_true = np.empty(num_nodes, dtype=int)
    fold_accuracies = []
    
    # For each fold: train on 9 folds, test on 1 fold (out-of-fold prediction)
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold+1}/10")
        train_mask = get_mask(train_idx, num_nodes)
        test_mask = get_mask(test_idx, num_nodes)

        model = GCN(num_features=data.num_node_features, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Training loop
        for epoch in tqdm(range(1, 201), desc=f"Training Fold {fold+1}"):
            loss = train(model, data, train_mask, optimizer)

        # Predict on the held-out fold
        acc, preds = test(model, data, test_mask)
        fold_accuracies.append(acc)
        print(f"Fold {fold+1} accuracy: {acc:.4f}")

        # Store predictions and ground truth for the test indices
        all_preds[test_idx] = preds
        all_true[test_idx] = data.y[test_idx].cpu().numpy()

    # Compute overall accuracy
    overall_acc = np.mean(all_preds == all_true)
    print(f"\nOverall out-of-fold accuracy: {overall_acc:.4f}")

    # Save predictions in the required format
    with open("predictions.tsv", "w") as f:
        for idx, pred in enumerate(all_preds):
            f.write(f"{idx}\t{class_names[pred]}\n")
    print("Predictions saved to predictions.tsv")

if __name__ == "__main__":
    main()
