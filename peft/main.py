from transformers import AutoModelForImageClassification
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

model_name_or_path = "microsoft/resnet-50"
#tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(inference_mode=False,
                         target_modules=["convolution"],
                         r=8,
                         lora_alpha=32,
                         lora_dropout=0.1)

model = AutoModelForImageClassification.from_pretrained(model_name_or_path)
for n, m in model.named_modules():
    print(n, type(m))
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282

from torch import nn
import torch

import torchvision.transforms as tf
from torchvision import datasets

transform = tf.Compose([tf.Resize([224, 224]), tf.ToTensor()])
datasets = datasets.ImageFolder('../pytorch/data', transform=transform)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(datasets, batch_size=16, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        #X = torch.tensor(X).to(torch.int64)
        #print(X)
        pred = model(X)
        pred = pred.logits
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help='give a number')
    args = parser.parse_args()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
        print("Done!")
