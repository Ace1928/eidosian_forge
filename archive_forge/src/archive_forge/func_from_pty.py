from __future__ import unicode_literals
from prompt_toolkit.filters import to_simple_filter, Condition
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from six.moves import range
import array
import errno
import os
import six
@classmethod
def from_pty(cls, stdout, true_color=False, ansi_colors_only=None, term=None):
    """
        Create an Output class from a pseudo terminal.
        (This will take the dimensions by reading the pseudo
        terminal attributes.)
        """
    assert stdout.isatty()

    def get_size():
        rows, columns = _get_size(stdout.fileno())
        return Size(rows=rows, columns=columns)
    return cls(stdout, get_size, true_color=true_color, ansi_colors_only=ansi_colors_only, term=term)