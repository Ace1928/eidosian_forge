from typing import Union
from warnings import warn
from .low_level import *
def new_signal(emitter, signal, signature=None, body=()):
    """Construct a new signal message

    :param DBusAddress emitter: The object sending the signal
    :param str signal: The name of the signal
    :param str signature: The DBus signature of the body data
    :param tuple body: Body data
    """
    header = new_header(MessageType.signal)
    header.fields[HeaderFields.path] = emitter.object_path
    if emitter.interface is None:
        raise ValueError('emitter.interface cannot be None for signals')
    header.fields[HeaderFields.interface] = emitter.interface
    header.fields[HeaderFields.member] = signal
    if signature is not None:
        header.fields[HeaderFields.signature] = signature
    return Message(header, body)