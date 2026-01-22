import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def bbox_to_corners(bbox) -> tuple:
    x0, top, x1, bottom = bbox
    return ((x0, top), (x0, bottom), (x1, top), (x1, bottom))