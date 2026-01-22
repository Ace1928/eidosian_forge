from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
@lru_cache(maxsize=2048)
def mess_ratio(decoded_sequence: str, maximum_threshold: float=0.2, debug: bool=False) -> float:
    """
    Compute a mess ratio given a decoded bytes sequence. The maximum threshold does stop the computation earlier.
    """
    detectors = [md_class() for md_class in MessDetectorPlugin.__subclasses__()]
    length = len(decoded_sequence) + 1
    mean_mess_ratio = 0.0
    if length < 512:
        intermediary_mean_mess_ratio_calc = 32
    elif length <= 1024:
        intermediary_mean_mess_ratio_calc = 64
    else:
        intermediary_mean_mess_ratio_calc = 128
    for character, index in zip(decoded_sequence + '\n', range(length)):
        for detector in detectors:
            if detector.eligible(character):
                detector.feed(character)
        if index > 0 and index % intermediary_mean_mess_ratio_calc == 0 or index == length - 1:
            mean_mess_ratio = sum((dt.ratio for dt in detectors))
            if mean_mess_ratio >= maximum_threshold:
                break
    if debug:
        for dt in detectors:
            print(dt.__class__, dt.ratio)
    return round(mean_mess_ratio, 3)