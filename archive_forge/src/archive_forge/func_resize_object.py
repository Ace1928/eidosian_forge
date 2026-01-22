import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def resize_object(obj, key: str, value):
    assert key in ('x0', 'x1', 'top', 'bottom')
    old_value = obj[key]
    diff = value - old_value
    new_items = [(key, value)]
    if key == 'x0':
        assert value <= obj['x1']
        new_items.append(('width', obj['x1'] - value))
    elif key == 'x1':
        assert value >= obj['x0']
        new_items.append(('width', value - obj['x0']))
    elif key == 'top':
        assert value <= obj['bottom']
        new_items.append(('doctop', obj['doctop'] + diff))
        new_items.append(('height', obj['height'] - diff))
        if 'y1' in obj:
            new_items.append(('y1', obj['y1'] - diff))
    elif key == 'bottom':
        assert value >= obj['top']
        new_items.append(('height', obj['height'] + diff))
        if 'y0' in obj:
            new_items.append(('y0', obj['y0'] - diff))
    return obj.__class__(tuple(obj.items()) + tuple(new_items))