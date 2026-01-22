from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def blocks_to_frames(blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int) -> Union[np.integer[Any], np.ndarray]:
    """Convert block indices to frame indices

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block

    Returns
    -------
    frames : np.ndarray [shape=samples.shape, dtype=int]
        The index or indices of frames corresponding to the beginning
        of each provided block.

    See Also
    --------
    blocks_to_samples
    blocks_to_time

    Examples
    --------
    Get frame indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_frame = librosa.blocks_to_frames(n, block_length=16)

    """
    return block_length * np.asanyarray(blocks)