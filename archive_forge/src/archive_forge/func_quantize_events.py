from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
def quantize_events(events, fps, length=None, shift=None):
    """
    Quantize the events with the given resolution.

    Parameters
    ----------
    events : list or numpy array
        Events to be quantized.
    fps : float
        Quantize with `fps` frames per second.
    length : int, optional
        Length of the returned array. If 'None', the length will be set
        according to the latest event.
    shift : float, optional
        Shift the events by `shift` seconds before quantization.

    Returns
    -------
    numpy array
        Quantized events.

    """
    events = np.array(events, dtype=np.float)
    if events.ndim != 1:
        raise ValueError('only 1-dimensional events supported.')
    if shift is not None:
        import warnings
        warnings.warn('`shift` parameter is deprecated as of version 0.16 and will be removed in version 0.18. Please shift the events manually before calling this function.')
        events += shift
    if length is None:
        length = int(round(np.max(events) * float(fps))) + 1
    else:
        events = events[:np.searchsorted(events, float(length - 0.5) / fps)]
    quantized = np.zeros(length)
    events *= fps
    idx = np.unique(np.round(events).astype(np.int))
    quantized[idx] = 1
    return quantized