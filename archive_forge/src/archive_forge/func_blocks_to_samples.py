from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def blocks_to_samples(blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int, hop_length: int) -> Union[np.integer[Any], np.ndarray]:
    """Convert block indices to sample indices

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block
    hop_length : int > 0
        The number of samples to advance between frames

    Returns
    -------
    samples : np.ndarray [shape=samples.shape, dtype=int]
        The index or indices of samples corresponding to the beginning
        of each provided block.

        Note that these correspond to the *first* sample index in
        each block, and are not frame-centered.

    See Also
    --------
    blocks_to_frames
    blocks_to_time

    Examples
    --------
    Get sample indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_sample = librosa.blocks_to_samples(n, block_length=16,
    ...                                          hop_length=512)

    """
    frames = blocks_to_frames(blocks, block_length=block_length)
    return frames_to_samples(frames, hop_length=hop_length)