import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def fails_on(db, reason=None):
    return fails_if(db, reason)