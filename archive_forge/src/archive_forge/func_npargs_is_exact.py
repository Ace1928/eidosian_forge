from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def npargs_is_exact(npargs):
    """
  Return true if npargs has an exact specification
  """
    return isinstance(npargs, int)