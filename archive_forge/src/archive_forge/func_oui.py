from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
@property
def oui(self):
    """The OUI (Organisationally Unique Identifier) for this EUI."""
    if self._module == _eui48:
        return OUI(self.value >> 24)
    elif self._module == _eui64:
        return OUI(self.value >> 40)