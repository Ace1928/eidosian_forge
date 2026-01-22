from __future__ import unicode_literals
from past.builtins import basestring
from ._utils import basestring
from .nodes import (
@output_operator()
def global_args(stream, *args):
    """Add extra global command-line argument(s), e.g. ``-progress``.
    """
    return GlobalNode(stream, global_args.__name__, args).stream()