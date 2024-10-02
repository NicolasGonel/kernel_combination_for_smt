from sklearn.datasets import fetch_openml
from smt.kernels import Product
from scipy.io import loadmat
from smt.kernels import SquarSinExp
from smt.kernels import Kernel
from smt.kernels import Matern32
from smt.kernels import Matern52

from smt.kernels import PowExp
import numpy as np
from smt.surrogate_models import KRG

import pickle
import os
import matplotlib.pyplot as plt
co2 = fetch_openml(data_id=41187, as_frame=True)

import polars as pl

co2_data = pl.DataFrame(co2.frame[["year", "month", "day", "co2"]]).select(
    pl.date("year", "month", "day"), "co2"
)

co2_data = (
    co2_data.sort(by="date")
    .group_by_dynamic("date", every="1mo")
    .agg(pl.col("co2").mean())
    .drop_nulls()
)

X = co2_data.select(
    pl.col("date").dt.year() + pl.col("date").dt.month() / 12
).to_numpy()[:100]
y = co2_data["co2"].to_numpy()[:100]
X_smt=X.reshape(-1)[:100]



class Period(Kernel):
    def __call__(self,d,grad_ind=None,hess_ind=None,derivative_params=None):
        n=self.theta.shape[0]
        theta2=self.theta[:n//2]
        theta3=self.theta[n//2:]
        return np.atleast_2d(np.exp(-np.sum(theta2*np.sin(theta3*d)**2,axis=1))).T
class RBF(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        theta=self.theta.reshape(1, d.shape[1])
        #r=np.zeros((d.shape[0],1))
        r=np.exp(-np.sum(theta*d**2,axis=1))
        return np.atleast_2d(r).T
class Rat_quad(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        n=self.theta.shape[0]
        theta4=self.theta[:n//2]
        theta5=self.theta[n//2:]
        r3=(1+d**2/(2*theta4*theta5))**(-theta4)
        return r3
class LocalPeriod(Kernel):
    def __call__(self, d, grad_ind=None, hess_ind=None, derivative_params=None):
        n=self.theta.shape[0]
        theta1=self.theta[:n//6]
        theta2=self.theta[n//6:2*n//6]
        theta3=self.theta[2*n//6:3*n//6]
        theta4=self.theta[3*n//6:4*n//6]
        theta5=self.theta[4*n//6:5*n//6]
        theta6=self.theta[5*n//6:]
        r1=np.exp(-np.sum(theta1*d**2,axis=1))
        r2=np.exp(-np.sum(theta2*np.sin(theta3*d)**2,axis=1))
        r3=(1+d**2/(2*theta4*theta5))**(-theta4)
        r4=np.exp(-np.sum(theta6*d**2,axis=1))
        r=(np.atleast_2d(r4).T+r3+np.atleast_2d(r2).T*np.atleast_2d(r1).T)/3
        return np.atleast_2d(r).T


k_test=LocalPeriod([0.01,0.01,0.01,0.01,0.01,0.01])    

k=RBF([0.01])+Period([0.01,0.01])*RBF([0.01])+Rat_quad([0.01,0.01])




from smt.surrogate_models import KRG
import time
sm=KRG(corr=k, hyper_opt="Cobyla",n_start=50)
sm.set_training_values(X, y)
sm.train()
print(sm.corr)

print(sm.corr)
import datetime

import numpy as np
import matplotlib.pyplot as plt

today = datetime.datetime.now()
current_month = today.year + today.month / 12
X_test = np.linspace(start=1958, stop=current_month, num=1_000).reshape(-1, 1)
mean_y_pred=sm.predict_values(X_test)
s2 = sm.predict_variances(X_test)



plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")
plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")

plt.legend()
plt.xlabel("Year")
plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
_ = plt.title(
    "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
)
_, axs = plt.subplots(1)

# add a plot with variance

axs.plot(X, y,color="black", linestyle="dashed", label="Measurements")
axs.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
axs.fill_between(
    np.ravel(X_test),
    np.ravel(mean_y_pred - 3 * np.sqrt(s2)),
    np.ravel(mean_y_pred + 3 * np.sqrt(s2)),
    color="lightgrey",
)


axs.legend()
plt.show()

