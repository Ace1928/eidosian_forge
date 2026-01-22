import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def make_alias(name):

    @functools.wraps(getattr(cls, name))
    def method(self, *args, **kwargs):
        return getattr(self, name)(*args, **kwargs)
    return method