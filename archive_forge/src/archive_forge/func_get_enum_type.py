import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
@staticmethod
def get_enum_type(atomic_type):
    """Returns the type of the 'enum' member for a BaseType whose
        'type' is 'atomic_type'."""
    return Type(BaseType(atomic_type), None, 1, sys.maxsize)