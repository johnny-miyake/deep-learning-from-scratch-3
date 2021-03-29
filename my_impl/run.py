import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x0 = Variable(np.array([2,3,4]))
x1 = Variable(np.array([3]))
y = x0 + x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)
