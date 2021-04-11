import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pylab as plt

max_epoch = 300
batch_size = 30
hidden_size = 100
lr = 1.0

x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

log_loss = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
    log_loss.append(avg_loss)

data = np.random.randn(3000, 2)
result = model(data)

answers = F.softmax(result).data.argmax(axis=1)

markers = ['bo', 'ro', 'go']

for i, a in enumerate(answers):
    px = data[i][0]
    py = data[i][1]
    mk = markers[a]
    if abs(px) <= 1.0 and abs(py) <= 1.0:
        plt.plot(px, py, mk, markersize=20)

plt.show()
