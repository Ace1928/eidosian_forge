import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def set_sender_name_owner(self, new_name):
    self._sender_name_owner = new_name