# Subject-Classification-using-Graph-Neural-Network
Cora dataset node classification using Graph Neural networks


[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)


This repository implements a Graph Convolutional Network (GCN) to classify scientific papers in the Cora dataset. The solution strictly follows the task requirements of 10-fold cross-validation with out-of-fold predictions.

---

## 🧩 Table of Contents
- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🧠 Approach](#-approach)
- [📊 Results](#-results)
- [🔗 References](#-references)

---

## ✨ Features

✅ Graph Neural Network (GCN) modeling using PyTorch Geometric  
✅ 10-fold cross-validation with out-of-fold predictions  
✅ TSV prediction file as required  
✅ Modular and clean codebase

---

## 📁 Project Structure

## Project Structure
```
cora-ml-pipeline/
├── data/ # Auto-downloaded dataset
├── src/
│ ├── model.py # GCN model definition
│ ├── train.py # Training + cross-validation
├── predictions.tsv # Final predictions
├── requirements.txt # Dependency list
└── README.md # This documentation
```


---

## ⚙️ Installation

### 1. Clone the Repository
```bash


git clone https://github.com/your-username/Subject-Classification-using-Graph-Neural-Network.git
cd Subject-Classification-using-Graph-Neural-Network

```

### 2. Install Dependencies
## 🐍 Creating a Virtual Environment

It is recommended to use a virtual environment to isolate your project's dependencies.

### ✅ Steps (for macOS/Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### ✅ Steps (for Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
💡 Don't forget to activate the environment each time you work on the project!



### 3. Download Data(auto-triggered on first run)
```bash
python -m src/download.py
```   

---

## 🚀 Usage

🔁 Train with 10-Fold Cross-Validation
```bash
python src/train.py
```
- Trains GCN model using scikit-learn’s KFold

- Generates predictions for all 2708 papers

- Saves out-of-fold predictions to predictions.tsv

#### Sample Output:

```python-repl

Fold 1/10 accuracy: 0.8930
...
Overall out-of-fold accuracy: 0.8774
```
---

## Approach

### 1. Graph Neural Network Architecture
The model uses **message passing** to combine node features with neighborhood information:

**model.py**
```python
class GCN(torch.nn.Module):
    def forward(self, x, edge_index):
        # First message-passing layer
        x = self.conv1(x, edge_index)  # Aggregates neighbor features
        x = F.relu(x)
        # Second message-passing layer
        x = self.conv2(x, edge_index)  # Further refines features
        return x
````


**Key Mechanisms:**
- **Node Features:** 1433-dim binary word vectors from `cora.content`
- **Neighborhood Aggregation:** Citation links define which nodes exchange information
- **Layer-Wise Transformation:** Learnable weights combine self + neighbor features

### 2. 10-Fold Cross-Validation
Implemented using scikit-learn's `KFold`:

**train.py**
```python

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
```
***
KFold(n_splits=10) splits your data into 10 parts.

For each fold, you get train_idx (for training) and test_idx (for testing).

For each fold:

  - The model is trained on the 9 folds (train_idx).

  - The model is tested (evaluates and predicts) on the 1 remaining fold (test_idx).

This is repeated 10 times, so every data point is predicted exactly once, always by a model that did not see it during training.

All predictions are concatenated in all_preds (which is filled only for the test indices in each fold).

- fold_accuracies stores the accuracy for each test fold.

- overall_acc is the mean of these, which is a standard way to report cross-validated accuracy.
***

### 3. Workflow:
1. Split 2708 nodes into 10 equal folds
2. For each fold:
   - Train on 9 folds (2437 nodes)
   - Predict on 1 fold (271 nodes)
3. Concatenate all fold predictions
4. Calculate final accuracy across all nodes

---

## Results

| Metric               | Value  |
|----------------------|--------|
| Cross-Val Accuracy   | 87.7%  |


**Prediction File Format (`predictions.tsv`):**

- 0 Genetic_Algorithms
- 1 Neural_Networks
...
- 2707 Theory

---
### ✅ Summary Table

| Step in Script                  | Task Requirement Fulfilled                            |
|----------------------------     |-------------------------------------------------------|
| KFold splitting                 | 10-fold CV, each fold used as test once               |
| Model trained on train_idx      | Model sees only 9 folds per iteration                 |
| Model tested on test_idx        | Each datapoint predicted out-of-fold                  |
| all_preds filled per test_idx   | Concatenated predictions for every datapoint          |
| Save `predictions.tsv`          | Required output format                                |

---

## References
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io)
- [Original GCN Paper](https://arxiv.org/abs/1609.02907)

---
