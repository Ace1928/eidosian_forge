from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def is_valid_trailing_comment(token):
    """
  Return true if the token is a valid trailing comment
  """
    return token.type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT) and (not comment_is_tag(token))