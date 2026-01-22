from __future__ import absolute_import
import time
import msvcrt  # pylint: disable=import-error
import contextlib
from jinxed import win32  # pylint: disable=import-error
from .terminal import WINSZ
from .terminal import Terminal as _Terminal
def getch(self):
    """
        Read, decode, and return the next byte from the keyboard stream.

        :rtype: unicode
        :returns: a single unicode character, or ``u''`` if a multi-byte
            sequence has not yet been fully received.

        For versions of Windows 10.0.10586 and later, the console is expected
        to be in ENABLE_VIRTUAL_TERMINAL_INPUT mode and the default method is
        called.

        For older versions of Windows, msvcrt.getwch() is used. If the received
        character is ``\\x00`` or ``\\xe0``, the next character is
        automatically retrieved.
        """
    if win32.VTMODE_SUPPORTED:
        return super(Terminal, self).getch()
    rtn = msvcrt.getwch()
    if rtn in ('\x00', 'à'):
        rtn += msvcrt.getwch()
    return rtn