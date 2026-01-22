from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def resolve_exported_function(self, pid, modName, procName):
    """
        Resolves the exported DLL function for the given process.

        @type  pid: int
        @param pid: Process global ID.

        @type  modName: str
        @param modName: Name of the module that exports the function.

        @type  procName: str
        @param procName: Name of the exported function to resolve.

        @rtype:  int, None
        @return: On success, the address of the exported function.
            On failure, returns C{None}.
        """
    aProcess = self.system.get_process(pid)
    aModule = aProcess.get_module_by_name(modName)
    if not aModule:
        aProcess.scan_modules()
        aModule = aProcess.get_module_by_name(modName)
    if aModule:
        address = aModule.resolve(procName)
        return address
    return None