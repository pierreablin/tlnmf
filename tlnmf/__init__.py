"""Transform Learning - NMF"""
__version__ = '0.0'  # noqa

from .tl_nmf import tl_nmf  # noqa
from .utils import signal_to_frames, unitary_projection  # noqa

import numpy as np

np.seterr(all='raise')
