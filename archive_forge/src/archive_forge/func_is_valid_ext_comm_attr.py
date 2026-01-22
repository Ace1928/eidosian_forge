import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_ext_comm_attr(attr):
    """Validates *attr* as string representation of RT or SOO.

    Returns True if *attr* is as per our convention of RT or SOO, else
    False. Our convention is to represent RT/SOO is a string with format:
    *global_admin_part:local_admin_path*
    """
    if not isinstance(attr, str):
        return False
    tokens = attr.rsplit(':', 1)
    if len(tokens) != 2:
        return False
    try:
        if '.' in tokens[0]:
            if not is_valid_ipv4(tokens[0]):
                return False
        else:
            int(tokens[0])
        int(tokens[1])
    except (ValueError, socket.error):
        return False
    return True