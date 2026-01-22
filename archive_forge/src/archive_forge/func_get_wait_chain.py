from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def get_wait_chain(self):
    """
        @rtype:
            tuple of (
            list of L{win32.WaitChainNodeInfo} structures,
            bool)
        @return:
            Wait chain for the thread.
            The boolean indicates if there's a cycle in the chain (a deadlock).
        @raise AttributeError:
            This method is only suppported in Windows Vista and above.
        @see:
            U{http://msdn.microsoft.com/en-us/library/ms681622%28VS.85%29.aspx}
        """
    with win32.OpenThreadWaitChainSession() as hWct:
        return win32.GetThreadWaitChain(hWct, ThreadId=self.get_tid())