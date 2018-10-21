import numpy as np

from numpy.testing import assert_allclose
from tlnmf import unitary_projection, tl_nmf


def test_unitary_projection():
    rng = np.random.RandomState(0)
    N = 3
    M = rng.randn(N, N)
    U = unitary_projection(M)
    assert_allclose(np.dot(U, U.T), np.eye(N), atol=1e-10)


def test_orthogonal_output():
    rng = np.random.RandomState(0)
    M = 3
    N = 10
    K = 2
    Y = rng.randn(M, N)
    Phi_init = unitary_projection(rng.randn(M, M))
    Phi, W, H, Phi_i, obj_list =\
        tl_nmf(Y, K, Phi=Phi_init, max_iter=1, n_iter_optim=1, rng=rng)
    assert_allclose(Phi_i, Phi_init)
    assert_allclose(np.dot(Phi, Phi.T), np.eye(M), atol=1e-10)
    assert_allclose(np.sum(W, axis=0), 1.)
    assert(Phi.shape == (M, M))
    assert(W.shape == (M, K))
    assert(H.shape == (K, N))
