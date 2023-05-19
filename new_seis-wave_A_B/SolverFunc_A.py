#求解器A模块
import numpy as np
import pandas as pd

def SolverFunc_A(seis, W, mu1, maxiter, tol):
    p = 1
    A = np.dot(W.T, W)
    inib = np.dot(W.T, seis)
    r = inib.copy()
    for k in range(maxiter):
        r1 = r.copy()
        Q = mu1 * np.diag(1/((np.abs(r))**(2-p) + np.finfo(float).eps)) # p
        Matrix = A + Q
        G = np.linalg.solve(Matrix, W.T)
        r = np.linalg.solve(Matrix, inib)
        r2 = r.copy()
        if np.sum(np.abs(r2 - r1)) / np.sum(np.abs(r1) + np.abs(r2)) < 0.5 * tol:
            break
    return r
