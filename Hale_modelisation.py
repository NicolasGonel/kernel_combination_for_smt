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

nodes_tr, edges_in_tr, edges_tar_tr, globals_tr, senders, receivers, list_tse_raw = pickle.load(open(os.path.join("HALE_50_2st2m_unitloads_tr","HALE_50_2st2m_unitloads_tr.pickle"), "rb"))


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

k=RBF([0.01])*Period([0.01,0.01])


for i in range(1,2):
    for j in range(10,11):

        x_Hale=np.arange(0,nodes_tr.shape[2]//10,10)
        y_Hale=nodes_tr[i,j,x_Hale]
        sm=KRG(corr=k, hyper_opt="Cobyla",n_start=10)
        sm.set_training_values(x_Hale, y_Hale)
        sm.train()
        print(k.theta)

        X_test = np.arange(0,nodes_tr.shape[2],1)   
        mean_y_pred=sm.predict_values(X_test)

        plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.5, label="Gaussian process Period*RBF")

        plt.scatter(x_Hale,y_Hale,color="g",label="data")
        plt.plot(X_test,nodes_tr[i,j,X_test],color="black",linestyle="dashed",alpha=0.3,label=f"ref {i} {j}")
        plt.legend()

        plt.show()



