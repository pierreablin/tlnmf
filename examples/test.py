from time import time

import matplotlib.pyplot as plt
import numpy as np

import soundfile as sf

from tlnmf import tl_nmf, signal_to_frames


rng = np.random.RandomState(1)
# Read the song
song_adr = 'datasets/armstrong.wav'

signal, fs = sf.read(song_adr)

# Compute the frame matrix

window_size = 40e-3
Y = signal_to_frames(signal, fs, window_size)

# Apply TL_NMF

K = 10

t0 = time()

Phi, W_d, H_d, obj_list, Pi = tl_nmf(Y, K, Phi='dct', verbose=True,
                                     max_iter=200, rng=rng, n_iter_optim=0,
                                     tol=1e-4, regul=0, regul_type='sparse',
                                     nmf=True)


Phi, W, H, obj_list, Pi = tl_nmf(Y, K, W=W_d, H=H_d, Phi='random',
                                 verbose=True,
                                 max_iter=200, rng=rng, n_iter_optim=5,
                                 tol=1e-6, regul=0, regul_type='sparse',
                                 nmf=False)

fit_time = time() - t0


# Plot the convergence curve:

t = np.linspace(0, fit_time, len(obj_list))

plt.figure()
plt.plot(t, obj_list)
plt.xlabel('Time (sec.)')
plt.ylabel('Objective function')
plt.show()


# Plot the most important atoms:

X = np.dot(Phi, Y)
power = np.linalg.norm(X, axis=1)
# power = np.sum(np.abs(Phi - Pi), axis=1)
shape_to_plot = (3, 2)

n_atoms = np.prod(shape_to_plot)
idx_to_plot = np.argsort(power)[-n_atoms:]

f, ax = plt.subplots(*shape_to_plot)
f.suptitle('Learned atoms')
for axe, idx in zip(ax.ravel(), idx_to_plot):
    axe.plot(Pi[idx])
    axe.plot(Phi[idx])
    axe.grid(False)
plt.show()
