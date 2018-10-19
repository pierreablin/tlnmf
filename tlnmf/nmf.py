import numpy as np


eps = 1e-15


def update_nmf_sparse(V, W, H, V_hat, regul):
    ''' One step of NMF with L1 regularization
    The algorithm is detailed in:

        Cedric Fevotte and Jerome Idier
        "Algorithms for non-negative matrix factorization with the
        beta-divergence"
        Neural Computations, vol. 23, no. 9, pp. 2421–2456, 2011

    Parameters
    ----------
    V : array, shape (M, N)
        Target spectrogram

    W : array, shape (M, K)
        Current dictionnary

    H : array, shape (K, N)
        Current activations

    V_hat : array, shape (M, N)
        Current learned spectrogram. Equals dot(W, H)

    regul : float
        Regularization level
    '''
    Ve = V + eps
    V_he = V_hat + eps
    H = H * (np.dot(W.T, Ve * V_he ** -2.) /
             (np.dot(W.T, 1. / V_he) + regul)) ** 0.5
    V_hat = np.dot(W, H)
    V_he = V_hat + eps
    W *= (np.dot(Ve * V_he ** -2., H.T) /
          (np.dot(1. / V_he, H.T) + regul * np.sum(H, axis=1)))**0.5
    # Normalize
    W = W / np.sum(W, axis=0)
    return W, H


def update_nmf_smooth(V, W, H, V_hat, regul):
    ''' One step of NMF with smooth regularization
    The algorithm is detailed in:

        Cedric Fevotte
        "Majorization-minimization algorithm for smooth itakura-saito
        nonnegative matrix factorization"
        International Conference on Acoustics, Speech and Signal Processing,
        2011, pp. 1980–1983

    Parameters
    ----------
    V : array, shape (M, N)
        Target spectrogram

    W : array, shape (M, K)
        Current dictionnary

    H : array, shape (K, N)
        Current activations

    V_hat : array, shape (M, N)
        Current learned spectrogram. Equals dot(W, H)

    regul : float
        Regularization level
    '''
    K, N = H.shape
    Ve = V + eps
    V_he = V_hat + eps
    # Minimization in H
    Gn = np.dot(W.T, Ve * V_he ** -2.)
    Gp = np.dot(W.T, 1. / V_he)
    # H1
    Ht = H.copy()
    p2 = Gp[:, 0] + regul / H[:, 1]
    p1 = - regul
    p0 = -Gn[:, 0] * H[:, 0] ** 2
    H[:, 0] = (np.sqrt(p1 ** 2 - 4. * p2 * p0) - p1) / (2 * p2)
    # Middle
    for n in range(1, N - 1):
        H[:, n] =\
            np.sqrt((Gn[:, n] * Ht[:, n] ** 2 + regul * H[:, n-1]) /
                    (Gp[:, n] + regul / H[:, n+1]))
    # HN
    p2 = Gp[:, N-1]
    p1 = regul
    p0 = - (Gn[:, N-1] * Ht[:, N-1] ** 2 + regul * H[:, N-2])
    H[:, N-1] = (np.sqrt(p1 ** 2 - 4. * p2 * p0) - p1) / (2 * p2)

    # Minimization in W
    V_hat = np.dot(W, H)
    V_he = V_hat + eps
    W = W * np.dot(Ve * V_he ** -2., H.T) / np.dot(1. / V_he, H.T)
    W = W / np.sum(W, axis=0)
    return W, H
