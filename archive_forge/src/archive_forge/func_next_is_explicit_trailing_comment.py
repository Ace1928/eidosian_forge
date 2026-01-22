from __future__ import print_function
from __future__ import unicode_literals
import logging
from operator import itemgetter as _itemgetter
import re
import sys
from cmakelang import lex
from cmakelang.common import UserError, InternalError
def next_is_explicit_trailing_comment(config, tokens):
    """
  Return true if the next comment is an explicit trailing comment, false
  otherwise
  """
    regex = re.compile('^' + config.markup.explicit_trailing_pattern + '.*')
    for token in iter_syntactic_tokens(tokens):
        return is_comment_matching_pattern(token, regex)
    return False