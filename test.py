import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv("static/concrete_data.csv")
X = df.drop('concrete_compressive_strength', axis=1).values
scale = MinMaxScaler()
X = scale.fit_transform(X)
a = [355, 232, 56, 76, 200, 210, 150, 50]


a = [np.array(a)]

a=scale.transform(a)

print(a)