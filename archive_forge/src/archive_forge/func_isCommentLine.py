import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
def isCommentLine(line):
    return EasytrieveLexer._COMMENT_LINE_REGEX.match(lines[0]) is not None