# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
import numpy as np


eps = 1e-15


def is_div(A, B):
    '''
    Computes the IS divergence
    '''
    M, N = A.shape
    f = (A + eps) / (B + eps)
    return np.sum(f - np.log(f)) - M * N


def penalty(H, regul_type):
    if regul_type == 'sparse':
        return np.sum(H)
    else:
        _, N = H.shape
        return is_div(H[:, :N-1], H[:, 1:])
