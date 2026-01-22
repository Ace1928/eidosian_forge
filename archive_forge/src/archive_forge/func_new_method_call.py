from typing import Union
from warnings import warn
from .low_level import *
def new_method_call(remote_obj, method, signature=None, body=()):
    """Construct a new method call message

    This is a relatively low-level method. In many cases, this will be called
    from a :class:`MessageGenerator` subclass which provides a more convenient
    API.

    :param DBusAddress remote_obj: The object to call a method on
    :param str method: The name of the method to call
    :param str signature: The DBus signature of the body data
    :param tuple body: Body data (i.e. method parameters)
    """
    header = new_header(MessageType.method_call)
    header.fields[HeaderFields.path] = remote_obj.object_path
    if remote_obj.bus_name is None:
        raise ValueError('remote_obj.bus_name cannot be None for method calls')
    header.fields[HeaderFields.destination] = remote_obj.bus_name
    if remote_obj.interface is not None:
        header.fields[HeaderFields.interface] = remote_obj.interface
    header.fields[HeaderFields.member] = method
    if signature is not None:
        header.fields[HeaderFields.signature] = signature
    return Message(header, body)