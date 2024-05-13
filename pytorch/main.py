import torchvision.transforms as tf
from torchvision import datasets

transform = tf.Compose([tf.Resize([224, 224]), tf.ToTensor()])
datasets = datasets.ImageFolder('./data', transform=transform)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(datasets, batch_size=16, shuffle=True)

from torch import nn
import torch


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(64)
        self.dense2 = nn.Linear(64, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = nn.functional.relu(self.dense1(x))
        x = nn.functional.softmax(self.dense2(x))

        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help='give a number')
    args = parser.parse_args()

    #transfer learning
    import torchvision
    model_tl = torchvision.models.resnet18(pretrained=True)
    model_tl.fc = nn.LazyLinear(2)
    #model = NeuralNetwork()
    print(model_tl)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_tl.parameters(), lr=args.learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model_tl, loss_fn, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
    print("Done!")
