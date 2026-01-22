from __future__ import unicode_literals
import inspect
import sys
import math
import numbers
from future.utils import PY2, PY3, exec_
def oct(number):
    """oct(number) -> string

        Return the octal representation of an integer
        """
    return '0' + builtins.oct(number)[2:]