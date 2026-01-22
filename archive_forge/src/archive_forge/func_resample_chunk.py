import warnings
import numpy as np
from . import cysoxr
from .cysoxr import QQ, LQ, MQ, HQ, VHQ
from ._version import version as __version__
def resample_chunk(self, x, last=False):
    """ Resample chunk with streaming resampler

        Parameters
        ----------
        x : array_like
            Input array. Input can be 1D(mono) or 2D(frames, channels).
            If input is not `np.ndarray` or not dtype in constructor,
            it will be converted to `np.ndarray` with dtype setting.

        last : bool, optional
            Set True at end of input sequence.

        Returns
        -------
        np.ndarray
            Resampled data.
            Output is np.ndarray with same ndim with input.

        """
    if type(x) != np.ndarray or x.dtype != self._type:
        warnings.warn(_CONVERT_WARN_STR.format(self._type), DeprecationWarning, stacklevel=2)
        x = np.asarray(x, dtype=self._type)
    x = np.ascontiguousarray(x)
    if x.ndim == 1:
        y = self._cysoxr.process(x[:, np.newaxis], last)
        return np.squeeze(y, axis=1)
    elif x.ndim == 2:
        return self._cysoxr.process(x, last)
    else:
        raise ValueError('Input must be 1-D or 2-D array')