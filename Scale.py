import pandas as pd
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

df = pd.read_csv ('.\Machin_Learning\cars.csv')

X = df[['Weight','Volume']]
y = df['CO2']
scaledX = scale.fit_transform(X)

regrss = lm.LinearRegression()
regrss.fit(scaledX, y)
scaled = scale.fit_transform([[2300,1.3]])

predictedCO2 = regrss.predict([scaled[0]])

print (predictedCO2)
