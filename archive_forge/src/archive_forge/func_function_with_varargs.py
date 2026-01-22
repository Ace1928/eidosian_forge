from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def function_with_varargs(arg1, arg2, arg3=1, *varargs):
    """Function with varargs.

  Args:
    arg1: Position arg docstring.
    arg2: Position arg docstring.
    arg3: Flags docstring.
    *varargs: Accepts unlimited positional args.
  Returns:
    The unlimited positional args.
  """
    del arg1, arg2, arg3
    return varargs