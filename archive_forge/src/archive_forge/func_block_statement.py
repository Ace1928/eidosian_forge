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
def block_statement(f):
    """Decorates a function where the node is a statement with children."""

    @contextlib.wraps(f)
    def wrapped(self, node, *args, **kwargs):
        self.prefix(node, default=self._indent)
        f(self, node, *args, **kwargs)
        if hasattr(self, 'block_suffix'):
            last_child = ast_utils.get_last_child(node)
            if last_child and last_child.lineno != getattr(node, 'lineno', 0):
                indent = (fmt.get(last_child, 'prefix') or '\n').splitlines()[-1]
                self.block_suffix(node, indent)
        else:
            self.suffix(node, comment=True)
    return wrapped