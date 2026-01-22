import sys
import math
import array
from .utils import (
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration
@register_pydub_effect
def invert_phase(seg, channels=(1, 1)):
    """
    channels- specifies which channel (left or right) to reverse the phase of.
    Note that mono AudioSegments will become stereo.
    """
    if channels == (1, 1):
        inverted = audioop.mul(seg._data, seg.sample_width, -1.0)
        return seg._spawn(data=inverted)
    else:
        if seg.channels == 2:
            left, right = seg.split_to_mono()
        else:
            raise Exception("Can't implicitly convert an AudioSegment with " + str(seg.channels) + ' channels to stereo.')
        if channels == (1, 0):
            left = left.invert_phase()
        else:
            right = right.invert_phase()
        return seg.from_mono_audiosegments(left, right)