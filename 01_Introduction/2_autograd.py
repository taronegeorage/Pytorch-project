from __future__ import print_function
import torch as t

#set requires_grad, pytorch will invoke autograd automatically
x = t.ones(2, 2, requires_grad=True)
print(x)

y = x.sum()
#y = x[0, 0]*x[0, 0] + x[0,1] + 3*x[1,0] + 8
print(y.grad_fn)
y.backward()

# y = x.sum() = (x[0][0] + x[0][1] + x[1][0] + x[1][1])
# every gradient is 1
print(x.grad)

# grad is accumulated, for every time run backward, the grad will increase, so need to set to zeros before backward
#y.backward()
#print(x.grad)

#y.backward()
#print(x.grad)

#set zeros
#x.grad.data.zero_()
#print(x)

#y.backward()
#print(x.grad)