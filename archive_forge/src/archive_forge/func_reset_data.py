from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def reset_data(*args, **kwargs):
    _state[0] = _iterate_read_data(read_data)
    if handle.readline.side_effect == _state[1]:
        _state[1] = _readline_side_effect()
        handle.readline.side_effect = _state[1]
    return DEFAULT