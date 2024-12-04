import numpy as np
import numpy
from sklearn.metrics import r2_score
from numpy import random as rand
import matplotlib.pyplot as plt

x= rand.normal(3,1,100)
y= rand.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = np.poly1d(np.polyfit(train_x, train_y , 4))

x_line = np.linspace(0,6,100)

plt.scatter(train_x,train_y)
plt.plot(x_line, mymodel(x_line))
plt.show()

r2 = r2_score(test_y, mymodel(test_x))
print(r2)