from typing import Union
from warnings import warn
from .low_level import *
def new_error(parent_msg, error_name, signature=None, body=()):
    """Construct a new error response message

    :param Message parent_msg: The method call this is a reply to
    :param str error_name: The name of the error
    :param str signature: The DBus signature of the body data
    :param tuple body: Body data
    """
    header = new_header(MessageType.error)
    header.fields[HeaderFields.reply_serial] = parent_msg.header.serial
    header.fields[HeaderFields.error_name] = error_name
    sender = parent_msg.header.fields.get(HeaderFields.sender, None)
    if sender is not None:
        header.fields[HeaderFields.destination] = sender
    if signature is not None:
        header.fields[HeaderFields.signature] = signature
    return Message(header, body)