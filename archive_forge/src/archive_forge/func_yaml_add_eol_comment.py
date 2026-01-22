from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def yaml_add_eol_comment(self, comment, key=NoComment, column=None):
    """
        there is a problem as eol comments should start with ' #'
        (but at the beginning of the line the space doesn't have to be before
        the #. The column index is for the # mark
        """
    from .tokens import CommentToken
    from .error import CommentMark
    if column is None:
        try:
            column = self._yaml_get_column(key)
        except AttributeError:
            column = 0
    if comment[0] != '#':
        comment = '# ' + comment
    if column is None:
        if comment[0] == '#':
            comment = ' ' + comment
            column = 0
    start_mark = CommentMark(column)
    ct = [CommentToken(comment, start_mark, None), None]
    self._yaml_add_eol_comment(ct, key=key)