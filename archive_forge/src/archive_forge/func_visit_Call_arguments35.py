from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys
from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator
def visit_Call_arguments35(self, node):

    def arg_compare(a1, a2):
        """Old-style comparator for sorting args."""

        def is_arg(a):
            return not isinstance(a, (ast.keyword, ast.Starred))
        if is_arg(a1) and isinstance(a2, ast.keyword):
            return -1
        elif is_arg(a2) and isinstance(a1, ast.keyword):
            return 1

        def get_pos(a):
            if isinstance(a, ast.keyword):
                a = a.value
            return (getattr(a, 'lineno', None), getattr(a, 'col_offset', None))
        pos1 = get_pos(a1)
        pos2 = get_pos(a2)
        if None in pos1 or None in pos2:
            return 0
        return -1 if pos1 < pos2 else 0 if pos1 == pos2 else 1
    all_args = sorted(node.args + node.keywords, key=functools.cmp_to_key(arg_compare))
    for i, arg in enumerate(all_args):
        self.visit(arg)
        if arg is not all_args[-1]:
            self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
    return bool(all_args)