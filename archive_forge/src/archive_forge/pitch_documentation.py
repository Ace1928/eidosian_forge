import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
Check the feasibility of yin/pyin parameters against
    the following conditions:

    1. 0 < fmin < fmax <= sr/2
    2. frame_length - win_length - 1 > sr/fmax
    