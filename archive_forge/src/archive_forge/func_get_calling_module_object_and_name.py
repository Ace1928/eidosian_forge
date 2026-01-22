from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def get_calling_module_object_and_name():
    """Returns the module that's calling into this module.

  We generally use this function to get the name of the module calling a
  DEFINE_foo... function.

  Returns:
    The module object that called into this one.

  Raises:
    AssertionError: Raised when no calling module could be identified.
  """
    for depth in range(1, sys.getrecursionlimit()):
        globals_for_frame = sys._getframe(depth).f_globals
        module, module_name = get_module_object_and_name(globals_for_frame)
        if id(module) not in disclaim_module_ids and module_name is not None:
            return _ModuleObjectAndName(module, module_name)
    raise AssertionError('No module was found')