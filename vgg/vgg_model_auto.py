import os
import torch
from torch import nn

#dictionary
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGGModel_auto(nn.Module):
    def __init__(self, letter, channel_size, tensor_size):
        super().__init__()

        self.lst = cfgs[letter]
        self.channel_size = channel_size
        self.tensor_size = tensor_size

        self.layers = nn.ModuleList([])

        for num in self.lst:
            if num == "M":
                self.layers.append(nn.MaxPool2d(2, 2))
                self.tensor_size //= 2
                continue

            self.layers.append(nn.Conv2d(self.channel_size, num, 3, 1, 1))
            self.layers.append(nn.ReLU())
            self.channel_size = num

        self.flatten = nn.Flatten()

        self.fc_relu_stack = nn.Sequential(
            nn.Linear(self.channel_size*self.tensor_size*self.tensor_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc_relu_stack(x)
        x = self.softmax(x)
        return x


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = VGGModel_auto("B", 3, 224).to(device)

X = torch.rand(1, 3, 224, 224, device=device)
logits = model(X)
y_pred = logits.argmax(1)
print(f"Predicted class: {y_pred}")
