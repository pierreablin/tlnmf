__version__ = '0.0'  # noqa

from .tl_nmf import tl_nmf  # noqa
from .utils import signal_to_frames  # noqa

import numpy as np

np.seterr(all='raise')
