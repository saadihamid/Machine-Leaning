import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
y = np.array([100.2,90.345,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])

y_fun = np.poly1d(np.polyfit(x,y,8))
x_new = np.linspace(1,22,100)
#print(np.modf(y))
plt.scatter(x,y)
plt.plot(x_new, y_fun(x_new))
plt.show()
print(y_fun(x[0]))
print (r2_score(y, y_fun(x)))