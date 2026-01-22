import sys
import math
import array
from .utils import (
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration
def rms_at(frame_i):
    return seg.get_sample_slice(frame_i - look_frames, frame_i).rms