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
def visit_Call_arguments(self, node):

    def arg_location(tup):
        arg = tup[1]
        if isinstance(arg, ast.keyword):
            arg = arg.value
        return (getattr(arg, 'lineno', 0), getattr(arg, 'col_offset', 0))
    if node.starargs:
        sorted_keywords = sorted([(None, kw) for kw in node.keywords] + [('*', node.starargs)], key=arg_location)
    else:
        sorted_keywords = [(None, kw) for kw in node.keywords]
    all_args = [(None, n) for n in node.args] + sorted_keywords
    if node.kwargs:
        all_args.append(('**', node.kwargs))
    for i, (prefix, arg) in enumerate(all_args):
        if prefix is not None:
            self.attr(node, '%s_prefix' % prefix, [self.ws, prefix], default=prefix)
        self.visit(arg)
        if arg is not all_args[-1][1]:
            self.attr(node, 'comma_%d' % i, [self.ws, ',', self.ws], default=', ')
    return bool(all_args)