import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def objects_to_bbox(objects):
    """
    Given an iterable of objects, return the smallest bounding box that
    contains them all.
    """
    return merge_bboxes(map(bbox_getter, objects))