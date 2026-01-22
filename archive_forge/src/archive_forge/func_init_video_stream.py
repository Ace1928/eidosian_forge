from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
def init_video_stream(self, codec: str, *, fps: float=24, pixel_format: str=None, max_keyframe_interval: int=None, force_keyframes: bool=None) -> None:
    """Initialize a new video stream.

        This function adds a new video stream to the ImageResource using the
        selected encoder (codec), framerate, and colorspace.

        Parameters
        ----------
        codec : str
            The codec to use, e.g. ``"x264"`` or ``"vp9"``.
        fps : float
            The desired framerate of the video stream (frames per second).
        pixel_format : str
            The pixel format to use while encoding frames. If None (default) use
            the codec's default.
        max_keyframe_interval : int
            The maximum distance between two intra frames (I-frames). Also known
            as GOP size. If unspecified use the codec's default. Note that not
            every I-frame is a keyframe; see the notes for details.
        force_keyframes : bool
            If True, limit inter frames dependency to frames within the current
            keyframe interval (GOP), i.e., force every I-frame to be a keyframe.
            If unspecified, use the codec's default.

        Notes
        -----
        You can usually leave ``max_keyframe_interval`` and ``force_keyframes``
        at their default values, unless you try to generate seek-optimized video
        or have a similar specialist use-case. In this case, ``force_keyframes``
        controls the ability to seek to _every_ I-frame, and
        ``max_keyframe_interval`` controls how close to a random frame you can
        seek. Low values allow more fine-grained seek at the expense of
        file-size (and thus I/O performance).

        """
    stream = self._container.add_stream(codec, fps)
    stream.time_base = Fraction(1 / fps).limit_denominator(int(2 ** 16 - 1))
    if pixel_format is not None:
        stream.pix_fmt = pixel_format
    if max_keyframe_interval is not None:
        stream.gop_size = max_keyframe_interval
    if force_keyframes is not None:
        stream.closed_gop = force_keyframes
    self._video_stream = stream