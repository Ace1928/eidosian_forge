from __future__ import absolute_import
import re
import ctypes
from ctypes.wintypes import BOOL
from ctypes.wintypes import HWND
from ctypes.wintypes import DWORD
from ctypes.wintypes import WORD
from ctypes.wintypes import LONG
from ctypes.wintypes import ULONG
from ctypes.wintypes import HKEY
from ctypes.wintypes import BYTE
import serial
from serial.win32 import ULONG_PTR
from serial.tools import list_ports_common
def get_parent_serial_number(child_devinst, child_vid, child_pid, depth=0, last_serial_number=None):
    """ Get the serial number of the parent of a device.

    Args:
        child_devinst: The device instance handle to get the parent serial number of.
        child_vid: The vendor ID of the child device.
        child_pid: The product ID of the child device.
        depth: The current iteration depth of the USB device tree.
    """
    if depth > MAX_USB_DEVICE_TREE_TRAVERSAL_DEPTH:
        return '' if not last_serial_number else last_serial_number
    devinst = DWORD()
    ret = CM_Get_Parent(ctypes.byref(devinst), child_devinst, 0)
    if ret:
        win_error = CM_MapCrToWin32Err(DWORD(ret), DWORD(0))
        if win_error == ERROR_NOT_FOUND:
            return '' if not last_serial_number else last_serial_number
        raise ctypes.WinError(win_error)
    parentHardwareID = ctypes.create_unicode_buffer(250)
    ret = CM_Get_Device_IDW(devinst, parentHardwareID, ctypes.sizeof(parentHardwareID) - 1, 0)
    if ret:
        raise ctypes.WinError(CM_MapCrToWin32Err(DWORD(ret), DWORD(0)))
    parentHardwareID_str = parentHardwareID.value
    m = re.search('VID_([0-9a-f]{4})(&PID_([0-9a-f]{4}))?(&MI_(\\d{2}))?(\\\\(.*))?', parentHardwareID_str, re.I)
    if not m:
        return '' if not last_serial_number else last_serial_number
    vid = None
    pid = None
    serial_number = None
    if m.group(1):
        vid = int(m.group(1), 16)
    if m.group(3):
        pid = int(m.group(3), 16)
    if m.group(7):
        serial_number = m.group(7)
    found_serial_number = serial_number
    if serial_number and (not re.match('^\\w+$', serial_number)):
        serial_number = None
    if not vid or not pid:
        return get_parent_serial_number(devinst, child_vid, child_pid, depth + 1, found_serial_number)
    if pid != child_pid or vid != child_vid:
        return '' if not last_serial_number else last_serial_number
    if not serial_number:
        return get_parent_serial_number(devinst, child_vid, child_pid, depth + 1, found_serial_number)
    return serial_number