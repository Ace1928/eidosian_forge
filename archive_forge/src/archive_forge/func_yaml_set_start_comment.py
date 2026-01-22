from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def yaml_set_start_comment(self, comment, indent=0):
    """overwrites any preceding comment lines on an object
        expects comment to be without `#` and possible have multiple lines
        """
    from .error import CommentMark
    from .tokens import CommentToken
    pre_comments = self._yaml_get_pre_comment()
    if comment[-1] == '\n':
        comment = comment[:-1]
    start_mark = CommentMark(indent)
    for com in comment.split('\n'):
        pre_comments.append(CommentToken('# ' + com + '\n', start_mark, None))