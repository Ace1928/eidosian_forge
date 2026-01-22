from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def comment_is_tag(token):
    """
  Return true if the comment token has one of the tag-forms:

  # cmake-format: <tag>
  # cmf: <tag>
  #[[cmake-format:<tag>]]
  #[[cmf:<tag>]]
  """
    return get_tag(token) is not None