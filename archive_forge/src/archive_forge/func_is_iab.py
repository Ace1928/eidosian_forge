from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
def is_iab(self):
    """:return: True if this EUI is an IAB address, False otherwise"""
    return self._value >> 24 in IAB.IAB_EUI_VALUES