import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

# def numerical_diff(f, x, eps=1e-4):
#     x0 = Variable(x.data - eps)
#     x1 = Variable(x.data + eps)
#     y0 = f(x0)
#     y1 = f(x1)
#     return (y1.data - y0.data) / (2 * eps)
# 
# 
# def square(x):
#     return Square()(x)
# 
# def exp(x):
#     return Exp()(x)
