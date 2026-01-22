from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import scipy.fftpack as fftpack
from ..processors import Processor
from .signal import Signal, FramedSignal
def local_group_delay(self, **kwargs):
    """
        Returns the local group delay of the phase.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to :class:`LocalGroupDelay`.

        Returns
        -------
        lgd : :class:`LocalGroupDelay` instance
            :class:`LocalGroupDelay` instance.

        """
    return LocalGroupDelay(self, **kwargs)