import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import _log_api_usage_once
from . import _video_opt
def read_video_timestamps(filename: str, pts_unit: str='pts') -> Tuple[List[int], Optional[float]]:
    """
    List the video frames timestamps.

    Note that the function decodes the whole video frame-by-frame.

    Args:
        filename (str): path to the video file
        pts_unit (str, optional): unit in which timestamp values will be returned
            either 'pts' or 'sec'. Defaults to 'pts'.

    Returns:
        pts (List[int] if pts_unit = 'pts', List[Fraction] if pts_unit = 'sec'):
            presentation timestamps for each one of the frames in the video.
        video_fps (float, optional): the frame rate for the video

    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(read_video_timestamps)
    from torchvision import get_video_backend
    if get_video_backend() != 'pyav':
        return _video_opt._read_video_timestamps(filename, pts_unit)
    _check_av_available()
    video_fps = None
    pts = []
    try:
        with av.open(filename, metadata_errors='ignore') as container:
            if container.streams.video:
                video_stream = container.streams.video[0]
                video_time_base = video_stream.time_base
                try:
                    pts = _decode_video_timestamps(container)
                except av.AVError:
                    warnings.warn(f'Failed decoding frames for file {filename}')
                video_fps = float(video_stream.average_rate)
    except av.AVError as e:
        msg = f'Failed to open container for {filename}; Caught error: {e}'
        warnings.warn(msg, RuntimeWarning)
    pts.sort()
    if pts_unit == 'sec':
        pts = [x * video_time_base for x in pts]
    return (pts, video_fps)