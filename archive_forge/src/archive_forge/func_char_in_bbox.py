import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def char_in_bbox(char, bbox) -> bool:
    v_mid = (char['top'] + char['bottom']) / 2
    h_mid = (char['x0'] + char['x1']) / 2
    x0, top, x1, bottom = bbox
    return bool(h_mid >= x0 and h_mid < x1 and (v_mid >= top) and (v_mid < bottom))