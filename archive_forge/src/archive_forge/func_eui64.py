from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
def eui64(self):
    """
        - If this object represents an EUI-48 it is converted to EUI-64             as per the standard.
        - If this object is already an EUI-64, a new, numerically             equivalent object is returned instead.

        :return: The value of this EUI object as a new 64-bit EUI object.
        """
    if self.version == 48:
        first_three = self._value >> 24
        last_three = self._value & 16777215
        new_value = first_three << 40 | 1099478073344 | last_three
    else:
        new_value = self._value
    return self.__class__(new_value, version=64)