import numpy as np
from utils import no_grad
from functions import *
from variable import Variable


a = Variable(np.array(3.0))
b = Variable(np.array(2.0))

print(a - b)
print(a / b)
print(a ** 3)
