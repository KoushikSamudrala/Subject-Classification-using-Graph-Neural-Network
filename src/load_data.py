from torch_geometric.datasets import Planetoid

def load_cora():
    dataset = Planetoid(root='data/raw', name='Cora')
    return dataset[0]

if __name__ == "__main__":
    data = load_cora()
    print(data)