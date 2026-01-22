import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classmethod
def clear_bp(cls, ctx, register):
    """
        Clears a hardware breakpoint.

        @see: find_slot, set_bp

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @type  register: int
        @param register: Slot (debug register) for hardware breakpoint.
        """
    ctx['Dr7'] &= cls.clearMask[register]
    ctx['Dr%d' % register] = 0