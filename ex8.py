import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

#data and model
data = load_iris()
model = LogisticRegression()

#whats in my data
# print(dir(data))
# print(data['data'])

#train test split
X_tr, X_te, Y_tr, Y_te = train_test_split(data.data, data.target, test_size=0.2)

#model fit and score
model.fit(X_tr,Y_tr)
print((model.score(X_te, Y_te)))

# print(len(X_tr), len(Y_te))
#confusion matrix
Y_pred = model.predict(X_te)

cm = confusion_matrix(Y_te, Y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True)
plt.xlabel('Pred')
plt.ylabel('Real')
plt.show()

