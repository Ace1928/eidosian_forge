import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
def connect_to_signal(self, signal_name, handler_function, dbus_interface=None, **keywords):
    """Arrange for a function to be called when the given signal is
        emitted.

        The parameters and keyword arguments are the same as for
        `dbus.proxies.ProxyObject.connect_to_signal`, except that if
        `dbus_interface` is None (the default), the D-Bus interface that
        was passed to the `Interface` constructor is used.
        """
    if not dbus_interface:
        dbus_interface = self._dbus_interface
    return self._obj.connect_to_signal(signal_name, handler_function, dbus_interface, **keywords)