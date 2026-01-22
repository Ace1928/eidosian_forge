import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def start_service_by_name(self, bus_name, flags=0):
    """Start a service which will implement the given bus name on this Bus.

        :Parameters:
            `bus_name` : str
                The well-known bus name to be activated.
            `flags` : dbus.UInt32
                Flags to pass to StartServiceByName (currently none are
                defined)

        :Returns: A tuple of 2 elements. The first is always True, the
            second is either START_REPLY_SUCCESS or
            START_REPLY_ALREADY_RUNNING.

        :Raises `DBusException`: if the service could not be started.
        :Since: 0.80.0
        """
    validate_bus_name(bus_name)
    return (True, self.call_blocking(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'StartServiceByName', 'su', (bus_name, flags)))