from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import numpy as np
def segment_axis(signal, frame_size, hop_size, axis=None, end='cut', end_value=0):
    """
    Generate a new array that chops the given array along the given axis into
    (overlapping) frames.

    Parameters
    ----------
    signal : numpy array
        Signal.
    frame_size : int
        Size of each frame [samples].
    hop_size : int
        Hop size between adjacent frames [samples].
    axis : int, optional
        Axis to operate on; if 'None', operate on the flattened array.
    end : {'cut', 'wrap', 'pad'}, optional
        What to do with the last frame, if the array is not evenly divisible
        into pieces; possible values:

        - 'cut'
          simply discard the extra values,
        - 'wrap'
          copy values from the beginning of the array,
        - 'pad'
          pad with a constant value.

    end_value : float, optional
        Value used to pad if `end` is 'pad'.

    Returns
    -------
    numpy array, shape (num_frames, frame_size)
        Array with overlapping frames

    Notes
    -----
    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    The returned array is always of type np.ndarray.

    Examples
    --------
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    """
    frame_size = int(frame_size)
    hop_size = int(hop_size)
    if axis is None:
        signal = np.ravel(signal)
        axis = 0
    if axis != 0:
        raise ValueError('please check if the resulting array is correct.')
    length = signal.shape[axis]
    if hop_size <= 0:
        raise ValueError('hop_size must be positive.')
    if frame_size <= 0:
        raise ValueError('frame_size must be positive.')
    if length < frame_size or (length - frame_size) % hop_size:
        if length > frame_size:
            round_up = frame_size + (1 + (length - frame_size) // hop_size) * hop_size
            round_down = frame_size + (length - frame_size) // hop_size * hop_size
        else:
            round_up = frame_size
            round_down = 0
        assert round_down < length < round_up
        assert round_up == round_down + hop_size or (round_up == frame_size and round_down == 0)
        signal = signal.swapaxes(-1, axis)
        if end == 'cut':
            signal = signal[..., :round_down]
        elif end in ['pad', 'wrap']:
            s = list(signal.shape)
            s[-1] = round_up
            y = np.empty(s, dtype=signal.dtype)
            y[..., :length] = signal
            if end == 'pad':
                y[..., length:] = end_value
            elif end == 'wrap':
                y[..., length:] = signal[..., :round_up - length]
            signal = y
        signal = signal.swapaxes(-1, axis)
    length = signal.shape[axis]
    if length == 0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; try end='pad' or end='wrap'")
    assert length >= frame_size
    assert (length - frame_size) % hop_size == 0
    n = 1 + (length - frame_size) // hop_size
    s = signal.strides[axis]
    new_shape = signal.shape[:axis] + (n, frame_size) + signal.shape[axis + 1:]
    new_strides = signal.strides[:axis] + (hop_size * s, s) + signal.strides[axis + 1:]
    try:
        return np.ndarray.__new__(np.ndarray, strides=new_strides, shape=new_shape, buffer=signal, dtype=signal.dtype)
    except TypeError:
        import warnings
        warnings.warn('Problem with ndarray creation forces copy.')
        signal = signal.copy()
        new_strides = signal.strides[:axis] + (hop_size * s, s) + signal.strides[axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=new_strides, shape=new_shape, buffer=signal, dtype=signal.dtype)