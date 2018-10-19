# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
import numbers

import numpy as np
from scipy import signal as sig


eps = 1e-15


def check_random_state(seed):
    '''Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def unitary_projection(M):
    ''' Projects M on the rotation manifold
    Parameters
    ----------
    M : array, shape (M, M)
        Input matrix
    '''
    s, u = np.linalg.eigh(np.dot(M, M.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), M)


def analysis_windowing(signal, M, Nbox):
    t = np.linspace(0, np.pi, num=M).reshape(M, 1)
    w = np.sin(t)
    Xbig = np.concatenate((np.zeros(M // 2), signal,
                           np.zeros(M // 2)), axis=0)
    Y = np.zeros((M, 2 * Nbox + 1))
    for kk in range(1, 2 * Nbox + 2):
        Y[:, kk - 1, np.newaxis] = w * \
            Xbig[np.arange(0, M) + (kk - 1) * (M // 2), np.newaxis]
    return Y


def synthesis_windowing(Y, M, Nbox):
    t = np.linspace(0, np.pi, num=M).reshape(M, 1)
    w = np.sin(t)
    X = np.dot(np.diag(w[:, 0]), Y)
    sX = np.zeros((int(M // 2), 2 * Nbox))

    for ll in range(1, 2 * Nbox + 1):
        sX[:, ll - 1, np.newaxis] =\
            X[M // 2:M, ll - 1, np.newaxis] + X[0:M // 2, ll, np.newaxis]

    x = (sX.T).reshape(sX.size, 1)
    return x


def signal_to_frames(signal, fs, window_size, fs_desired=None):
    ''' From a 1-d sound signal, builds the corresponding frame matrix Y
    Parameters
    ----------
    signal : array, shape (n_samples, )
        Input sound signal

    fs : float
        Sampling frequency of fs

    window_size : float
        Window size in ms

    fs_desired: float | None
        Over/sub-samble the signal.
    '''
    if fs_desired is not None:
        signal = sig.resample_poly(signal, fs_desired, fs)
        fs = fs_desired
    n_samples = len(signal)
    n_window_samples = 2 * np.floor(window_size * fs / 2)
    N_box = int(np.ceil(n_samples / n_window_samples))
    n_window_samples = int(n_window_samples)
    padding = np.zeros(n_window_samples * N_box)
    signal = np.concatenate((signal, padding))

    return analysis_windowing(signal, n_window_samples, N_box)
