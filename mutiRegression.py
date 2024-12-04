import scipy
import sklearn
import pandas as pd
from sklearn import linear_model as lm

df = pd.read_csv('.\Machin_Learning\cars.csv')
X= df[['Weight','Volume']]
y= df['CO2']
#print(X)
regr = lm.LinearRegression()
regr.fit(X,y)
predicted_CO2 = regr.predict([[2300,1300]])
print (predicted_CO2)
print(regr.coef_)