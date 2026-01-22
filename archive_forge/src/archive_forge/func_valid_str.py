import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def valid_str(addr):
    """
    :param addr: An IEEE EUI-64 identifier in string form.

    :return: ``True`` if EUI-64 identifier is valid, ``False`` otherwise.
    """
    try:
        if _get_match_result(addr, RE_EUI64_FORMATS):
            return True
    except TypeError:
        pass
    return False