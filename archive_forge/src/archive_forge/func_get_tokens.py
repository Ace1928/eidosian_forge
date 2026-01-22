from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import common
from cmakelang import lex
from cmakelang.parse.util import (
def get_tokens(self, out=None, kind=None):
    if out is None:
        out = []
    if kind is None:
        kind = 'all'
    match_group = {'semantic': is_semantic_token, 'syntactic': is_syntactic_token, 'whitespace': is_whitespace_token, 'comment': is_comment_token, 'all': lambda x: True}[kind]
    for child in self.children:
        if isinstance(child, lex.Token):
            if match_group(child):
                out.append(child)
        elif isinstance(child, TreeNode):
            child.get_tokens(out, kind)
        else:
            raise RuntimeError('Unexpected child of type {}'.format(type(child)))
    return out