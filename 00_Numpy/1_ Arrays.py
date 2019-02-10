import numpy as np

a = np.array([1,2,3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 99
print(a)

b = np.array([[1,2],[3,4],[5,6]])
print(b.shape)
print(b[0], b[1], b[2])
print(b[0,0], b[0,1], b[2,0])


print('other functions')
a = np.zeros(3)
a = np.zeros((3,4))
print(a)

b = np.ones((2,2))
print(b)

c = np.full((3,3), 4)
print(c)

d = np.eye(3)
print(d)

e = np.random.random((2,2))
print(e)