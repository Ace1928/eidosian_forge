from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.compat import utf8, unichr, PY3, check_anchorname_char, nprint  # NOQA
def fetch_comment(self, comment):
    value, start_mark, end_mark = comment
    while value and value[-1] == ' ':
        value = value[:-1]
    self.tokens.append(CommentToken(value, start_mark, end_mark))