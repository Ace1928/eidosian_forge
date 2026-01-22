import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
def get_clip_location(self, idx: int) -> Tuple[int, int]:
    """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
    video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    if video_idx == 0:
        clip_idx = idx
    else:
        clip_idx = idx - self.cumulative_sizes[video_idx - 1]
    return (video_idx, clip_idx)