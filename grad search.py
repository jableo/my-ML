import numpy as np
import pandas as pd

data = pd.read_csv("test_scores.csv")

def grad_search(x,y):
    m_curr= b_curr=0
    iters = 10**6
    n = len(x)
    learning_rate= 10**-5
    for i in range(iters):

        y_predicted = m_curr * x +b_curr
        md = -(2/n)*sum(x*(y - y_predicted))
        bd = -(2/n)*sum(y - y_predicted)
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m: {}, b: {}, iteration: {}, cost: {}".format(m_curr,b_curr,i, cost))


xs = np.array(data['math']).copy()
ys = np.array(data['cs']).copy()
# xs = np.array([1,5,0.3,0.8,1.3])
# ys = np.array([5,21,2.2,4.2,6.2])

grad_search(xs, ys)
# print(xs, ys)

# print(data['math'].to_numpy(), data['cs'].to_numpy())