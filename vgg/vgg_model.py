import os
import torch
from torch import nn

class VGGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#224->112

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#112->56

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#56->28

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#28->14

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),#14->7
        )
        self.flatten = nn.Flatten()
        self.fc_relu_stack = nn.Sequential(
            nn.Linear(512*49, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_relu_stack(x)
        print(f"size of conv tensor: {x.size()}")
        x = self.flatten(x)
        print(f"size of flatten tensor: {x.size()}")
        x = self.fc_relu_stack(x)
        print(f"size of fc tensor: {x.size()}")
        print(x)
        x = self.softmax(x)
        print(f"size of softmax tensor: {x.size()}")
        print(x)
        return x

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")
model = VGGModel().to(device)

X = torch.rand(1, 3, 224, 224, device=device)
logits = model(X)
y_pred = logits.argmax(1)
print(f"Predicted class: {y_pred}")
