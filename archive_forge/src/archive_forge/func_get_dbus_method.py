import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
def get_dbus_method(self, member, dbus_interface=None):
    """Return a proxy method representing the given D-Bus method.

        This is the same as `dbus.proxies.ProxyObject.get_dbus_method`
        except that if `dbus_interface` is None (the default),
        the D-Bus interface that was passed to the `Interface` constructor
        is used.
        """
    if dbus_interface is None:
        dbus_interface = self._dbus_interface
    return self._obj.get_dbus_method(member, dbus_interface)