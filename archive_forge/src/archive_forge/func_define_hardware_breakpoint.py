from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def define_hardware_breakpoint(self, dwThreadId, address, triggerFlag=BP_BREAK_ON_ACCESS, sizeFlag=BP_WATCH_DWORD, condition=True, action=None):
    """
        Creates a disabled hardware breakpoint at the given address.

        @see:
            L{has_hardware_breakpoint},
            L{get_hardware_breakpoint},
            L{enable_hardware_breakpoint},
            L{enable_one_shot_hardware_breakpoint},
            L{disable_hardware_breakpoint},
            L{erase_hardware_breakpoint}

        @note:
            Hardware breakpoints do not seem to work properly on VirtualBox.
            See U{http://www.virtualbox.org/ticket/477}.

        @type  dwThreadId: int
        @param dwThreadId: Thread global ID.

        @type  address: int
        @param address: Memory address to watch.

        @type  triggerFlag: int
        @param triggerFlag: Trigger of breakpoint. Must be one of the following:

             - L{BP_BREAK_ON_EXECUTION}

               Break on code execution.

             - L{BP_BREAK_ON_WRITE}

               Break on memory read or write.

             - L{BP_BREAK_ON_ACCESS}

               Break on memory write.

        @type  sizeFlag: int
        @param sizeFlag: Size of breakpoint. Must be one of the following:

             - L{BP_WATCH_BYTE}

               One (1) byte in size.

             - L{BP_WATCH_WORD}

               Two (2) bytes in size.

             - L{BP_WATCH_DWORD}

               Four (4) bytes in size.

             - L{BP_WATCH_QWORD}

               Eight (8) bytes in size.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

            The callback signature is::

                def condition_callback(event):
                    return True     # returns True or False

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @type  action: function
        @param action: (Optional) Action callback function.
            If specified, the event is handled by this callback instead of
            being dispatched normally.

            The callback signature is::

                def action_callback(event):
                    pass        # no return value

            Where B{event} is an L{Event} object,
            and the return value is a boolean
            (C{True} to dispatch the event, C{False} otherwise).

        @rtype:  L{HardwareBreakpoint}
        @return: The hardware breakpoint object.
        """
    thread = self.system.get_thread(dwThreadId)
    bp = HardwareBreakpoint(address, triggerFlag, sizeFlag, condition, action)
    begin = bp.get_address()
    end = begin + bp.get_size()
    if dwThreadId in self.__hardwareBP:
        bpSet = self.__hardwareBP[dwThreadId]
        for oldbp in bpSet:
            old_begin = oldbp.get_address()
            old_end = old_begin + oldbp.get_size()
            if MemoryAddresses.do_ranges_intersect(begin, end, old_begin, old_end):
                msg = 'Already exists (TID %d) : %r' % (dwThreadId, oldbp)
                raise KeyError(msg)
    else:
        bpSet = set()
        self.__hardwareBP[dwThreadId] = bpSet
    bpSet.add(bp)
    return bp