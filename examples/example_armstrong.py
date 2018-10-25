# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
from time import time

import matplotlib.pyplot as plt
import numpy as np

import soundfile as sf

from tlnmf import tl_nmf, signal_to_frames


rng = np.random.RandomState(0)
# Read the song
song_adr = 'datasets/armstrong.wav'

signal, fs = sf.read(song_adr)

# Compute the frame matrix

window_size = 40e-3
Y = signal_to_frames(signal, fs, window_size)

# Apply TL_NMF

K = 10

t0 = time()

Phi, W, H, Phi_init, infos = tl_nmf(Y, K, verbose=True, rng=rng)

fit_time = time() - t0


# Plot the convergence curve:
obj_list = infos['obj_list']
t = np.linspace(0, fit_time, len(obj_list))

plt.figure()
plt.loglog(t, obj_list)
plt.xlabel('Time (sec.)')
plt.ylabel('Objective function')
plt.show()


# Plot the most important atoms:

X = np.dot(Phi, Y)
power = np.linalg.norm(X, axis=1)
shape_to_plot = (3, 2)

n_atoms = np.prod(shape_to_plot)
idx_to_plot = np.argsort(power)[-n_atoms:]

f, ax = plt.subplots(*shape_to_plot)
f.suptitle('Learned atoms')
for axe, idx in zip(ax.ravel(), idx_to_plot):
    axe.plot(Phi[idx])
    axe.axis('off')
plt.show()
