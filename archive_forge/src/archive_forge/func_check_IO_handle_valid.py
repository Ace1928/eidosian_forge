import abc
from os_brick import exception
from os_brick import executor
from os_brick import initiator
def check_IO_handle_valid(self, handle, data_type, protocol):
    """Check IO handle has correct data type."""
    if handle and (not isinstance(handle, data_type)):
        raise exception.InvalidIOHandleObject(protocol=protocol, actual_type=type(handle))