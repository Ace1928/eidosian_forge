import sys
import math
import array
from .utils import (
from .silence import split_on_silence
from .exceptions import TooManyMissingFrames, InvalidDuration
def db_over_threshold(rms):
    if rms == 0:
        return 0.0
    db = ratio_to_db(rms / thresh_rms)
    return max(db, 0)