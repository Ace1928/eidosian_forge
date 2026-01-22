from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def search_for_locationID_in_interfaces(serial_interfaces, locationID):
    for interface in serial_interfaces:
        if interface.id == locationID:
            return interface.name
    return None