import time
import weakref
import threading
import pyglet
from pyglet.libs.win32 import com
from pyglet.event import EventDispatcher
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _ole32 as ole32, _oleaut32 as oleaut32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.input.base import Device, Controller, Button, AbsoluteAxis, ControllerManager
def get_xinput_guids():
    """We iterate over all devices in the system looking for IG_ in the device ID, which indicates it's an
    XInput device. Returns a list of strings containing pid/vid.
    Monstrosity found at: https://docs.microsoft.com/en-us/windows/win32/xinput/xinput-and-directinput
    """
    guids_found = []
    locator = IWbemLocator()
    services = IWbemServices()
    enum_devices = IEnumWbemClassObject()
    devices = (IWbemClassObject * 20)()
    ole32.CoCreateInstance(CLSID_WbemLocator, None, CLSCTX_INPROC_SERVER, IID_IWbemLocator, byref(locator))
    name_space = BSTR('\\\\.\\root\\cimv2')
    class_name = BSTR('Win32_PNPEntity')
    device_id = BSTR('DeviceID')
    hr = locator.ConnectServer(name_space, None, None, 0, 0, None, None, byref(services))
    if hr != 0:
        return guids_found
    hr = ole32.CoSetProxyBlanket(services, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, None, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, None, EOAC_NONE)
    if hr != 0:
        return guids_found
    hr = services.CreateInstanceEnum(class_name, 0, None, byref(enum_devices))
    if hr != 0:
        return guids_found
    var = VARIANT()
    oleaut32.VariantInit(byref(var))
    while True:
        returned = ULONG()
        _hr = enum_devices.Next(10000, len(devices), devices, byref(returned))
        if returned.value == 0:
            break
        for i in range(returned.value):
            result = devices[i].Get(device_id, 0, byref(var), None, None)
            if result == 0:
                if var.vt == VT_BSTR and var.bstrVal != '':
                    if 'IG_' in var.bstrVal:
                        guid = var.bstrVal
                        pid_start = guid.index('PID_') + 4
                        dev_pid = guid[pid_start:pid_start + 4]
                        vid_start = guid.index('VID_') + 4
                        dev_vid = guid[vid_start:vid_start + 4]
                        sdl_guid = f'{dev_pid}{dev_vid}'.lower()
                        if sdl_guid not in guids_found:
                            guids_found.append(sdl_guid)
    oleaut32.VariantClear(byref(var))
    return guids_found