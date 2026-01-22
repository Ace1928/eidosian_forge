from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@staticmethod
def get_active_services():
    """
        Retrieve a list of all active system services.

        @see: L{get_services},
            L{start_service}, L{stop_service},
            L{pause_service}, L{resume_service}

        @rtype:  list( L{win32.ServiceStatusProcessEntry} )
        @return: List of service status descriptors.
        """
    with win32.OpenSCManager(dwDesiredAccess=win32.SC_MANAGER_ENUMERATE_SERVICE) as hSCManager:
        return [entry for entry in win32.EnumServicesStatusEx(hSCManager, dwServiceType=win32.SERVICE_WIN32, dwServiceState=win32.SERVICE_ACTIVE) if entry.ProcessId]