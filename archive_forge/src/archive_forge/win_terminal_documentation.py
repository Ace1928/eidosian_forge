from __future__ import absolute_import
import time
import msvcrt  # pylint: disable=import-error
import contextlib
from jinxed import win32  # pylint: disable=import-error
from .terminal import WINSZ
from .terminal import Terminal as _Terminal

        A context manager for ``jinxed.w32.setcbreak()``.

        Although both :meth:`break` and :meth:`raw` modes allow each keystroke
        to be read immediately after it is pressed, Raw mode disables
        processing of input and output.

        In cbreak mode, special input characters such as ``^C`` are
        interpreted by the terminal driver and excluded from the stdin stream.
        In raw mode these values are receive by the :meth:`inkey` method.
        