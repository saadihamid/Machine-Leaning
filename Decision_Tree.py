from os import P_DETACH
import pandas as pd
from pydotplus import graphviz
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier as dtreeclassi
import matplotlib.image as pltimg
import matplotlib.pyplot as plt
import pydotplus 

df = pd.read_csv('.\Machin_Learning\shows.csv')
d={'UK':0, 'USA': 1, 'N':2}
df['Nationality'] = df['Nationality'].map(d)
d={'NO':0,'YES':1}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

dtree = dtreeclassi()
dtree =dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)

graph = pydotplus.graph_from_dot_data(data)
graph.write_png('myDecitionTree.png')

img = pltimg.imread('myDecitionTree.png', format='png')
plt.imshow(img)
plt.show()
