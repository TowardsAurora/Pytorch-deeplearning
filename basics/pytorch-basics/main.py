import torch
import torch.nn as nn


x = torch.randn(10,3)
y = torch.randn(10,2)

# print(x)
# print(y)
# Build a fully connected layer.
linear =   nn.Linear(3,2)
w = linear.weight
b = linear.bias

print ('w: ', w)
print ('b: ', b)

criterion = nn.MSELoss()
optimizer= torch.optim.SGD(linear.parameters(),lr = 0.01)
for epoch in range(100):

# Forward pass.
    y_pred = linear(x)

    # Compute the loss.
    loss = criterion(y_pred, y)
    print(loss.item())
    loss.backward()
    #
    # print("dL/dw:",  linear.weight.grad)
    # print("dL/db",linear.bias.grad)

    optimizer.step()

    pred = linear(x)
    loss = criterion(pred, y)
    print("After {} optimizer step, loss:".format(epoch), loss.item())