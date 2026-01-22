import os
import re
def find_system_bus():
    addr = os.environ.get('DBUS_SYSTEM_BUS_ADDRESS', '') or 'unix:path=/var/run/dbus/system_bus_socket'
    return next(get_connectable_addresses(addr))