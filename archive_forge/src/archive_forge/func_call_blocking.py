import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def call_blocking(self, bus_name, object_path, dbus_interface, method, signature, args, timeout=-1.0, byte_arrays=False, **kwargs):
    """Call the given method, synchronously.
        :Since: 0.81.0
        """
    if object_path == LOCAL_PATH:
        raise DBusException('Methods may not be called on the reserved path %s' % LOCAL_PATH)
    if dbus_interface == LOCAL_IFACE:
        raise DBusException('Methods may not be called on the reserved interface %s' % LOCAL_IFACE)
    get_args_opts = dict(byte_arrays=byte_arrays)
    if 'utf8_strings' in kwargs:
        raise TypeError("unexpected keyword argument 'utf8_strings'")
    message = MethodCallMessage(destination=bus_name, path=object_path, interface=dbus_interface, method=method)
    try:
        message.append(*args, signature=signature)
    except Exception as e:
        logging.basicConfig()
        _logger.error('Unable to set arguments %r according to signature %r: %s: %s', args, signature, e.__class__, e)
        raise
    reply_message = self.send_message_with_reply_and_block(message, timeout)
    args_list = reply_message.get_args_list(**get_args_opts)
    if len(args_list) == 0:
        return None
    elif len(args_list) == 1:
        return args_list[0]
    else:
        return tuple(args_list)