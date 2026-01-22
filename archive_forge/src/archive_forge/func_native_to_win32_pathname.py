import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def native_to_win32_pathname(name):
    """
        @type  name: str
        @param name: Native (NT) absolute pathname.

        @rtype:  str
        @return: Win32 absolute pathname.
        """
    if name.startswith('\\'):
        if name.startswith('\\??\\'):
            name = name[4:]
        elif name.startswith('\\SystemRoot\\'):
            system_root_path = os.environ['SYSTEMROOT']
            if system_root_path.endswith('\\'):
                system_root_path = system_root_path[:-1]
            name = system_root_path + name[11:]
        else:
            for drive_number in compat.xrange(ord('A'), ord('Z') + 1):
                drive_letter = '%c:' % drive_number
                try:
                    device_native_path = win32.QueryDosDevice(drive_letter)
                except WindowsError:
                    e = sys.exc_info()[1]
                    if e.winerror in (win32.ERROR_FILE_NOT_FOUND, win32.ERROR_PATH_NOT_FOUND):
                        continue
                    raise
                if not device_native_path.endswith('\\'):
                    device_native_path += '\\'
                if name.startswith(device_native_path):
                    name = drive_letter + '\\' + name[len(device_native_path):]
                    break
    return name