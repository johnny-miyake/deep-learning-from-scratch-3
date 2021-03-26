import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.transpose(x)
print(x.T)
print(y)
y.backward(retain_grad=True)
print(x.grad)