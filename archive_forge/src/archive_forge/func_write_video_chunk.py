from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
import torch
import torio
def write_video_chunk(self, i: int, chunk: torch.Tensor, pts: Optional[float]=None):
    """Write video/image data

        Args:
            i (int): Stream index.
            chunk (Tensor): Video/image tensor.
                Shape: `(time, channel, height, width)`.
                The ``dtype`` must be ``torch.uint8``.
                The shape (height, width and the number of channels) must match
                what was configured when calling :py:meth:`add_video_stream`
            pts (float, optional or None): If provided, overwrite the presentation timestamp.

                .. note::

                   The provided value is converted to integer value expressed in basis of
                   frame rate. Therefore, it is truncated to the nearest value of
                   ``n / frame_rate``.
        """
    self._s.write_video_chunk(i, chunk, pts)