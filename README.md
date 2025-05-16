# Subject-Classification-using-Graph-Neural-Network
Cora dataset node classification using Graph Neural networks


[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a Graph Convolutional Network (GCN) to classify scientific papers in the Cora dataset. The solution strictly follows the task requirements of 10-fold cross-validation with out-of-fold predictions.

![GCN Architecture](https://miro.medium.com/v2/resize:fit:1400/1*ZkF2QH4Bx4H2aEEkK0E6AA.png)

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Approach](#approach)
- [Results](#results)
- [References](#references)

---

## Features
✅ 10-fold cross-validation with out-of-fold predictions  
✅ Graph Neural Network (GCN) implementation  
✅ Complete reproducibility (seed fixed)  
✅ TSV prediction file generation  
✅ Modular codebase

---

## Project Structure

cora-ml-pipeline/
├── data/ # Auto-downloaded dataset
├── src/
│ ├── model.py # GCN model definition
│ ├── train.py # Training + cross-validation
│ └── infer.py # Inference with saved model
├── predictions.tsv # Final predictions
├── requirements.txt # Dependency list
└── README.md # This documentation

text

---

## Installation

1. **Clone Repository**
git clone https://github.com/your-username/cora-ml-pipeline.git
cd cora-ml-pipeline

text

2. **Install Dependencies**
pip install -r requirements.txt

text

3. **Download Data** *(auto-triggered on first run)*

---

## Usage

### Training with Cross-Validation
python src/train.py

text
- Performs 10-fold CV
- Saves predictions to `predictions.tsv`
- Output:
Fold 1/10 accuracy: 0.8143
...
Overall accuracy: 0.8120



## Approach

### 1. Graph Neural Network Architecture
The model uses **message passing** to combine node features with neighborhood information:

model.py
class GCN(torch.nn.Module):
def forward(self, x, edge_index):
# First message-passing layer
x = self.conv1(x, edge_index) # Aggregates neighbor features
x = F.relu(x)
# Second message-passing layer
x = self.conv2(x, edge_index) # Further refines features
return x

text

**Key Mechanisms:**
- **Node Features:** 1433-dim binary word vectors from `cora.content`
- **Neighborhood Aggregation:** Citation links define which nodes exchange information
- **Layer-Wise Transformation:** Learnable weights combine self + neighbor features

### 2. 10-Fold Cross-Validation
Implemented using scikit-learn's `KFold`:

train.py
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
# Train on 90% of nodes
model = GCN(...)
# Test on remaining 10%
acc, preds = test(model, data, test_mask)
# Store out-of-fold predictions
all_preds[test_idx] = preds

text

**Workflow:**
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
| Cross-Val Accuracy   | 81.2%  |
| Inference Time       | <1 sec |

**Prediction File Format (`predictions.tsv`):**
0 Genetic_Algorithms
1 Neural_Networks
...
2707 Theory

text

---

## References
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io)
- [Original GCN Paper](https://arxiv.org/abs/1609.02907)
- [Cora Dataset Paper](https://people.cs.umass.edu/~mccallum/papers/cora.pdf)

---