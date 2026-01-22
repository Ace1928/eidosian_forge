import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_moref_type(moref):
    """Get the type of a managed object reference

    This function accepts a string representation of a ManagedObjectReference
    like `VirtualMachine:vm-123`, but is also able to extract it from the
    actual object as returned by the API.
    """
    if isinstance(moref, str):
        if ':' in moref:
            splits = moref.split(':')
            return splits[0]
        return None
    return moref._type