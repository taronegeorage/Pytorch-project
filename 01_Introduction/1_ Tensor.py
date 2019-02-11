from __future__ import print_function
import torch as t

print('Pytorch vision:', t.__version__)

#construct 5x3 matrix, not initialized
x = t.Tensor(5, 3)

x = t.Tensor([[1,2],[3,4]])
print(x)
print(type(x))

#random
x = t.rand(4, 3)
print(x)

print(x.size())
print(x.size(1), x.size()[1])

y = t.rand(4, 3)
#add 1
print(x+y)
#add 2
print(t.add(x,y))
#or
result = t.Tensor(4,3)
t.add(x,y,out=result)
print(result)

#inplace change the value of y
y.add_(x)

print('slice Tensor')
print(x)
print(x[:, 1])

print('Tensor <-> Numpy')
a = t.ones(5)
b = a.numpy()
print(a, b)

import numpy as np
a = np.ones(5)
b = t.from_numpy(a)
print(a, b)
#Tensor and Numpy share the memery
b.add_(1)
print(a, b)

#withdraw
scalar = b[0]
print(scalar)
print(scalar.item()) #python scalars

tensor = t.tensor([2])
print(tensor,scalar)
print(tensor.size(),scalar.size())
print(tensor.item(), scalar.item())

print('torch.tensor')
tensor = t.tensor([3,4])
scalar = t.tensor(3)
print(scalar)

#not share memeory
old_tensor = tensor
new_tensor = t.tensor(old_tensor)
new_tensor[0] = 1111
print(old_tensor, new_tensor)
#torch.from_numpy() or tensor.detach() share
new_tensor = old_tensor.detach()
new_tensor[0] = 1111
print(old_tensor, new_tensor)

print('GPU tensor')
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
z = x+y
