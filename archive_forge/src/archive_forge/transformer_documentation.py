import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
Applies a function to each individual assignment.

    This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
    It tries to break down the unpacking if possible. In effect, it has the same
    effect as passing the assigned values in SSA form to apply_fn.

    Examples:

    The following will result in apply_fn(a, c), apply_fn(b, d):

        a, b = c, d

    The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

        a, b = c

    The following will result in apply_fn(a, (b, c)):

        a = b, c

    It uses the visitor pattern to allow subclasses to process single
    assignments individually.

    Args:
      targets: list, tuple of or individual AST node. Should be used with the
        targets field of an ast.Assign node.
      values: an AST node.
      apply_fn: a function of a single argument, which will be called with the
        respective nodes of each single assignment. The signature is
        apply_fn(target, value), no return value.
    