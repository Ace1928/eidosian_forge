from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
@property
def writing(self):
    """`bool`: `True` if the mode permits writing."""
    return 'w' in self or 'a' in self or '+' in self or ('x' in self)