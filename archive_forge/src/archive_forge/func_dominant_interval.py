from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
def dominant_interval(self, histogram):
    """
        Extract the dominant interval of the given histogram.

        Parameters
        ----------
        histogram : tuple
            Histogram (tuple of 2 numpy arrays, the first giving the strengths
            of the bins and the second corresponding delay values).

        Returns
        -------
        interval : int
            Dominant interval.

        """
    return dominant_interval(histogram, self.hist_smooth)