from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
import torch
import torio
def write_audio_chunk(self, i: int, chunk: torch.Tensor, pts: Optional[float]=None):
    """Write audio data

        Args:
            i (int): Stream index.
            chunk (Tensor): Waveform tensor. Shape: `(frame, channel)`.
                The ``dtype`` must match what was passed to :py:meth:`add_audio_stream` method.
            pts (float, optional, or None): If provided, overwrite the presentation timestamp.

                .. note::

                   The provided value is converted to integer value expressed in basis of
                   sample rate. Therefore, it is truncated to the nearest value of
                   ``n / sample_rate``.
        """
    self._s.write_audio_chunk(i, chunk, pts)