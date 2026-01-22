import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def bbox_to_rect(bbox) -> dict:
    """
    Return the rectangle (i.e a dict with keys "x0", "top", "x1",
    "bottom") for an object.
    """
    return {'x0': bbox[0], 'top': bbox[1], 'x1': bbox[2], 'bottom': bbox[3]}