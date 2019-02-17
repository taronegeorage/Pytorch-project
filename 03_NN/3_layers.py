from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
from torch import nn
import matplotlib.pyplot as plt

to_tensor = ToTensor()
to_pil = ToPILImage()
img = Image.open('/Users/taronegeorage/Desktop/test.png')

input = to_tensor(img).unsqueeze(0)

# Convolution layer: eg. ruihua
kernel = t.ones(3, 3) / -9.
kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)

out = conv(input)
plt.imshow(to_pil(out.data.squeeze(0)))
plt.axis('off') 
plt.show()

# Pooling layer
pool = nn.AvgPool2d(2,2)
out = pool(input)
plt.imshow(to_pil(out.data.squeeze(0)))
plt.axis('off') 
plt.show()

# Linear (FC) layer, BN, Dropout
input = t.randn(2, 3)
linear = nn.Linear(3, 4)
h = linear(input)

bn = nn.BatchNorm1d(4)
bn.weight.data = t.ones(4) * 4
bn.bias.data = t.zeros(4)
bn_out = bn(h)
print(h)
print(bn_out)
print(bn_out.mean(0), bn_out.var(0, unbiased=False))

dropout = nn.Dropout(0.5)
o = dropout(bn_out)
print(o)


# activation funvtion
relu = nn.ReLU(inplace=True)
