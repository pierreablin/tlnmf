# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
import numpy as np
from scipy.optimize import line_search
from scipy.linalg import expm

eps = 1e-15


def fast_transform_learning(Phi, X, V_hat, n_iter_optim, n_ls_tries=30):
    '''Optimize w.r.t Phi
    '''
    M, _ = Phi.shape
    obj = None
    V_he = V_hat + eps
    for n in range(n_iter_optim):
        # Gradient
        Xe = X + eps
        Xe_sq = Xe ** 2
        V_hat_inv = 1. / V_he
        G = 2 * np.dot(Xe * (V_hat_inv - 1. / Xe_sq), X.T)
        G = 0.5 * (G - G.T)  # Project
        # Hessian
        H = 2 * np.dot(1. / Xe_sq + V_hat_inv, Xe_sq.T)
        # Project
        H = 0.5 * (H + H.T)
        # Search direction:
        E = - G / H
        # Line-search
        transform, X_new, converged, obj =\
            line_search_scipy(Xe, E, G, V_he, obj, n_ls_tries)
        if not converged:
            print('break')
            break
        Phi = np.dot(transform, Phi)
        X[:] = X_new
    return Phi, X


def line_search_scipy(X, E, G, V_hat, current_loss, n_ls_tries):
    M, _ = E.shape

    class function_caller(object):
        def __init__(self):
            pass

        def myf(self, E):
            self.transform = expm(E.reshape(M, M))
            new_X = np.dot(self.transform, X)
            new_X += eps
            self.new_X = np.dot(self.transform, X)
            f = new_X ** 2 / V_hat
            return np.sum(f - np.log(f))

        def myfprime(self, E):
            new_X = self.new_X
            G = 2 * np.dot(new_X / V_hat - 1. / new_X, X.T)
            return 0.5 * (G - G.T).ravel()

    fc = function_caller()
    xk = np.zeros(M ** 2)
    gfk = G.ravel()
    pk = E.ravel()
    old_fval = current_loss
    alpha, _, _, new_fval, _, _ = line_search(fc.myf, fc.myfprime, xk, pk, gfk,
                                              old_fval, maxiter=n_ls_tries)
    if alpha is not None:
        transform = fc.transform
        X_new = fc.new_X
        obj = new_fval
        return transform, X_new, True, obj
    else:
        return 0, 0, False, 0
