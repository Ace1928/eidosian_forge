import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def msg_reply_handler(message):
    if isinstance(message, MethodReturnMessage):
        reply_handler(*message.get_args_list(**get_args_opts))
    elif isinstance(message, ErrorMessage):
        error_handler(DBusException(*message.get_args_list(), name=message.get_error_name()))
    else:
        error_handler(TypeError('Unexpected type for reply message: %r' % message))