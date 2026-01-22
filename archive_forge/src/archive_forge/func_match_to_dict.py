import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def match_to_dict(self, m, main_group: int=0, return_groups: bool=True, return_chars: bool=True) -> dict:
    subset = self.tuples[m.start(main_group):m.end(main_group)]
    chars = [c for text, c in subset if c is not None]
    x0, top, x1, bottom = objects_to_bbox(chars)
    result = {'text': m.group(main_group), 'x0': x0, 'top': top, 'x1': x1, 'bottom': bottom}
    if return_groups:
        result['groups'] = m.groups()
    if return_chars:
        result['chars'] = chars
    return result