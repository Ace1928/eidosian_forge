import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def fails_if(predicate, reason=None):
    rule = compound()
    pred = _as_predicate(predicate, reason)
    rule.fails.add(pred)
    return rule