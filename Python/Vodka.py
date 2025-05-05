import torch
import torch.nn as nn
import torch.optim as optim

# Example XOR dataset
inputs = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
targets = torch.tensor([[0.],[1.],[1.],[0.]])

# Define a simple neural net
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(1000):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss {loss.item()}")

# Test output
print(model(inputs))
