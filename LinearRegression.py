import pandas as pd
import numpy as np

df = pd.read_csv("coffee_shop_revenue.csv")

#gan vector X va Y voi tap du lieu
x = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values.reshape(-1, 1)

#chuan hoa cac bien dau vao
x[:,0]/=500
x[:,1]/=10
x[:,2]/=18
x[:,3]/=15
x[:,4]/=500
x[:,5]/=1000

#them cot bias va init w
col1 = np.ones((x.shape[0], 1))
X = np.concatenate((col1, x), axis=1)
w = np.array([[1],[2],[3],[4],[5],[6],[7]])

#ham loss
def MSE(w):
    return 0.5 * np.mean((y - X.dot(w))**2)

#Tinh dao ham ham loss
def Gradient (w):
    N = X.shape[0]
    return (1/N)*(X.T.dot(X.dot(w) - y))

#Thuat toan GradientDescent
def gradientDescent(w, LR):
    for i in range(100000):
        tmp_w = w - LR*Gradient(w)
        if MSE(tmp_w) < e :
            return tmp_w
        w = tmp_w
    return w

#Tim w
e = 1e-4
LR = 0.01
w = gradientDescent(w,LR)
print("Trong so w sau khi toi uu la: w = \n",w)

#Du doan
x_new = np.array([[1],[112],[3.6],[10],[3],[456.4],[234]])
print("y du doan se la: y_predict = ",x_new.T.dot(w))