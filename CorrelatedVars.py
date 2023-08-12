import numpy as np
import matplotlib.pyplot as plt

rho = -0.7
Ndraws = 1000
mu = np.array([0,0])
cov = np.array([[1, rho] , [rho , 1]])

W = np.random.multivariate_normal(mu, cov, size=Ndraws)

plt.plot(W.cumsum(axis=0));
plt.title('Correlated Random Variables')
plt.show()
print(np.corrcoef(W.T))

variable_1 = W[:, 0].cumsum(axis=0)
variable_2 = W[:, 1].cumsum(axis=0)
if Ndraws < 20:
    print(f"varialble 1:{variable_1}")
    print(f"varialble 2:{variable_2}")