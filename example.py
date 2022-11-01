import torch
from torch import nn

from rbf import RBFLayer

# 3-class classification
# data and labels
data = [
    [0.75, 1.0],
    [0.5, 0.75],
    [0.25, 0.0],
    [0.5, 0.0],
    [0.0, 0.0],
    [1.0, 0.75],
    [1.0, 1.0],
    [0.5, 0.25],
    [0.75, 0.5],
]

labels = [
    [1, -1, -1],
    [1, -1, -1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, 1],
    [-1, -1, 1],
]


class RBFClassification(nn.Module):
    """
    RBF Network for classification
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(RBFClassification, self).__init__()
        self.rbf = RBFLayer(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.rbf(x)
        out = self.fc(out)
        return out


def train(model, data, labels, lr, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            pred = outputs.data.argmax(dim=1)
            gt = labels.data.argmax(dim=1)
            acc = (pred == gt).sum().item() / labels.size(0)
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Accuracy: {acc*100}%")
            if acc == 1:
                break


def main():
    print("-------RBF Network-------")
    model = RBFClassification(2, 4, 3)
    train(model, torch.Tensor(data), torch.Tensor(labels), lr=0.01, epochs=100000)


if __name__ == "__main__":
    main()
