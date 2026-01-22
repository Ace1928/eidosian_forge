import hashlib
import os
from enum import Enum, auto
def get_store_name(group_name):
    """Generate the unique name for the NCCLUniqueID store (named actor).

    Args:
        group_name: unique user name for the store.
    Return:
        str: MD5-hexlified name for the store.
    """
    if not group_name:
        raise ValueError('group_name is None.')
    hexlified_name = hashlib.md5(group_name.encode()).hexdigest()
    return hexlified_name