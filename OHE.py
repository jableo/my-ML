import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
model = LinearRegression()
le = LabelEncoder()
cars = pd.read_csv('carprices.csv')
ct = ColumnTransformer([("Car Model", OneHotEncoder(), [1])], remainder="passthrough")
nefle = cars
nefle["Car Model"]= le.fit_transform(nefle["Car Model"])
print(nefle)
X = nefle[["Mileage","Car Model","Age(yrs)"]].values
Y = nefle["Sell Price($)"].values
X = ct.fit_transform(X)
X = X[:, 1:]
print(X)
model.fit(X,Y)
print(model.score(X, Y))
print(model.predict([[0,1,45000, 4]]))
print(model.predict([[1,0,86000, 7]]))
