from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events

        Add onset peak-picking related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        threshold : float
            Threshold for peak-picking.
        smooth : float, optional
            Smooth the activation function over `smooth` seconds.
        pre_avg : float, optional
            Use `pre_avg` seconds past information for moving average.
        post_avg : float, optional
            Use `post_avg` seconds future information for moving average.
        pre_max : float, optional
            Use `pre_max` seconds past information for moving maximum.
        post_max : float, optional
            Use `post_max` seconds future information for moving maximum.
        combine : float, optional
            Only report one onset within `combine` seconds.
        delay : float, optional
            Report the detected onsets `delay` seconds delayed.

        Returns
        -------
        parser_group : argparse argument group
            Onset peak-picking argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        