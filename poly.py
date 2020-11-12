import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#job rank, 1 being the lowest and 10 being the higest 
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

#annual salary in rands
y = [[45000], [50000], [60000], [80000], [110000], [150000], [200000], [300000], [500000], [700000]]

#set degree of polynomial regression model to 4  
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)

#train and test regressor quadratic model   
polreg= LinearRegression()
polreg.fit(x_poly,y)

#train and test linear regression model
regressor = LinearRegression()
regressor.fit(x, y)
xx = np.linspace(0, 10, 4)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))


#plot data
plt.axis([0,10,0,1e6])
plt.scatter(x,y,c='b')
plt.plot(xx, yy)
plt.plot(x,polreg.predict(poly.fit_transform(x)),c='red',linestyle='--')
plt.xlabel("Employee Rank")
plt.ylabel("Salary in Rands(R)")
plt.title("Salary regressed on job rank")
plt.grid(True)


plt.show()

print(x)
print(x_poly)
