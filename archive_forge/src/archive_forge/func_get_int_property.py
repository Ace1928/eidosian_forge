from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def get_int_property(device_type, property, cf_number_type):
    """
    Search the given device for the specified string property

    @param device_type Device to search
    @param property String to search for
    @param cf_number_type CFType number

    @return Python string containing the value, or None if not found.
    """
    key = cf.CFStringCreateWithCString(kCFAllocatorDefault, property.encode('utf-8'), kCFStringEncodingUTF8)
    CFContainer = iokit.IORegistryEntryCreateCFProperty(device_type, key, kCFAllocatorDefault, 0)
    if CFContainer:
        if cf_number_type == kCFNumberSInt32Type:
            number = ctypes.c_uint32()
        elif cf_number_type == kCFNumberSInt16Type:
            number = ctypes.c_uint16()
        cf.CFNumberGetValue(CFContainer, cf_number_type, ctypes.byref(number))
        cf.CFRelease(CFContainer)
        return number.value
    return None