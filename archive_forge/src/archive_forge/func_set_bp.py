import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classmethod
def set_bp(cls, ctx, register, address, trigger, watch):
    """
        Sets a hardware breakpoint.

        @see: clear_bp, find_slot

        @type  ctx: dict( str S{->} int )
        @param ctx: Thread context dictionary.

        @type  register: int
        @param register: Slot (debug register).

        @type  address: int
        @param address: Memory address.

        @type  trigger: int
        @param trigger: Trigger flag. See L{HardwareBreakpoint.validTriggers}.

        @type  watch: int
        @param watch: Watch flag. See L{HardwareBreakpoint.validWatchSizes}.
        """
    Dr7 = ctx['Dr7']
    Dr7 |= cls.enableMask[register]
    orMask, andMask = cls.triggerMask[register][trigger]
    Dr7 &= andMask
    Dr7 |= orMask
    orMask, andMask = cls.watchMask[register][watch]
    Dr7 &= andMask
    Dr7 |= orMask
    ctx['Dr7'] = Dr7
    ctx['Dr%d' % register] = address