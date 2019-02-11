from __future__ import print_function
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # execute super construct function
        # equivalent to nn.Module.__init__(self)
        super(Net, self).__init__()

        # convolution layer '1': 1 - single channel, 6 - output channel, 5 - kernel size 5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # convolution layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layer, y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape, '-1' - adaptation
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# parameters in the NN
params = list(net.parameters())
print(len(params))
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

# input and output of forward() are Tensor
input = t.rand(1, 1, 32, 32)
out = net(input)
print(out.size())

net.zero_grad()
out.backward(t.ones(1, 10))

# loss function
output = net(input)
target = t.arange(0,10).view(1,10) 
target = target.float()
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

#back propagate
net.zero_grad() 
print('conv1 bias gradient before backward()', net.conv1.bias.grad)
loss.backward()
print('conv1 bias gradient after backward()', net.conv1.bias.grad)

#optimizer
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad() 
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
