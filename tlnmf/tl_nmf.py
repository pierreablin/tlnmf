# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
import numpy as np
from scipy import fftpack

from .utils import check_random_state, unitary_projection
from .functions import is_div, penalty
from .nmf import update_nmf_smooth, update_nmf_sparse
from .transform_learning import fast_transform_learning


def tl_nmf(Y, K, Phi=None, W=None, H=None, regul=None, max_iter=300,
           n_iter_optim=5, tol=1e-5, verbose=False, rng=None):
    '''Runs Transform learning NMF

    Parameters
    ----------
    Y : array, shape (M, N)
        Frames matrix

    K : int
        Rank of the learned feature matrices.

    Phi : array, shape (M, M) | 'random' | 'dct' | None, optional
        Initial Transform. Should be orthogonal. If 'random', start from a
        random orthogonal matrix. If 'dct', start from the DCT coefficients.
        Random by default

    W : array, shape (M, K) | None, optional
        Initial dictionnary.

    H : array, shape (K, N) | None, optional
        Initial activations.

    regul : float | None, optional
        Level of regularization. By default, a heuristic is used.

    max_iter : int, optional
        Maximal number of iterations for the algorithm

    n_iter_optim : int, optional
        Number of iteration of Transform learning between NMF steps

    tol : float, optional
        tolerance for the stopping criterion. Iterations stop when two
        consecutive iterations of the algorithm have a relative objective
        change lower than tol.

    verbose : boolean, optional
        Wether to print or not informations about the current state

    rng : RandomState, optional
        random seed of the algorithm

    Returns
    -------
    Phi : array, shape (M, M)
        The estimated transform matrix

    W : array, shape (M, K)
        The estimated dictionnary

    H : array, shape (K, N)
        The estimated activations

    obj_list : list
        list of objective values
    '''
    eps = 1e-15
    regul_type = 'sparse'
    M, N = Y.shape

    rng = check_random_state(rng)
    # Initialization
    if regul is None:
        regul = 1e6
    if Phi is None:
        Phi = 'random'
    if Phi == 'random':
        Phi = unitary_projection(rng.randn(M, M))
    elif Phi == 'dct':
        Phi = fftpack.dct(np.eye(M), 3, norm='ortho')
    if W is None:
        W = np.abs(rng.randn(M, K)) + 1.
        W = W / np.sum(W, axis=0)
    if H is None:
        H = np.abs(rng.randn(K, N)) + 1.

    X = np.dot(Phi, Y)
    V = X ** 2  # Initial spectrogram
    V_hat = np.dot(W, H)  # Initial factorization

    obj = is_div(V, V_hat) + regul * penalty(H, regul_type)  # Objective
    obj_list = []
    Phi_init = Phi.copy()
    # Verbose
    if verbose:
        print('Running TL-NMF with %s regularization on a %d x %d '
              'problem with K = %d' % (regul_type, M, N, K))
        print(' | '.join([name.center(8) for name in
                         ["iter", "obj", "eps", "NMF", "TL", "d_phi",
                          "d_phi_i"]]))
    for n in range(max_iter):
        # NMF
        if regul_type == 'smooth':
            W, H = update_nmf_smooth(V, W, H, V_hat, regul)
        else:
            W, H = update_nmf_sparse(V, W, H, V_hat, regul)
        # Transform Learning
        V_hat = np.dot(W, H)
        obj1 = is_div(V, V_hat) + regul * penalty(H, regul_type)
        Phi_old = Phi.copy()
        Phi, X = fast_transform_learning(Phi, X, V_hat, n_iter_optim)
        V = X ** 2
        # Check terminaison
        old_obj = obj.copy()
        obj = is_div(V, V_hat) + regul * penalty(H, regul_type)
        obj_list.append(obj)
        eps = (old_obj - obj) / (np.abs(obj) + np.abs(old_obj))
        if np.abs(eps) < tol:
            break

        if verbose:
            eps1 = old_obj - obj1
            eps2 = obj1 - obj
            delta_phi = np.linalg.norm(Phi - Phi_old) / M ** 2
            delta_phi_init = np.linalg.norm(Phi - Phi_init) / M ** 2
            print(' | '.join([("%d" % (n+1)).rjust(8),
                              ("%.2e" % obj).rjust(8),
                              ("%.2e" % eps).rjust(8),
                              ("%.2e" % eps1).rjust(8),
                              ("%.2e" % eps2).rjust(8),
                              ("%.2e" % delta_phi).rjust(8),
                              ("%.2e" % delta_phi_init).rjust(8)]))
    return Phi, W, H, Phi_init, obj_list
