import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def obj_to_bbox(obj):
    """
    Return the bounding box for an object.
    """
    return bbox_getter(obj)