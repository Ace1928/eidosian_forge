import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
@property
def odd(self):
    return bool(self.index % 2)