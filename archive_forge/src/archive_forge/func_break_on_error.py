from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def break_on_error(self, pid, errorCode):
    """
        Sets or clears the system breakpoint for a given Win32 error code.

        Use L{Process.is_system_defined_breakpoint} to tell if a breakpoint
        exception was caused by a system breakpoint or by the application
        itself (for example because of a failed assertion in the code).

        @note: This functionality is only available since Windows Server 2003.
            In 2003 it only breaks on error values set externally to the
            kernel32.dll library, but this was fixed in Windows Vista.

        @warn: This method will fail if the debug symbols for ntdll (kernel32
            in Windows 2003) are not present. For more information see:
            L{System.fix_symbol_store_path}.

        @see: U{http://www.nynaeve.net/?p=147}

        @type  pid: int
        @param pid: Process ID.

        @type  errorCode: int
        @param errorCode: Win32 error code to stop on. Set to C{0} or
            C{ERROR_SUCCESS} to clear the breakpoint instead.

        @raise NotImplementedError:
            The functionality is not supported in this system.

        @raise WindowsError:
            An error occurred while processing this request.
        """
    aProcess = self.system.get_process(pid)
    address = aProcess.get_break_on_error_ptr()
    if not address:
        raise NotImplementedError('The functionality is not supported in this system.')
    aProcess.write_dword(address, errorCode)