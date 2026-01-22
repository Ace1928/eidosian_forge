from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
def ipv6_link_local(self):
    """
        .. note:: This poses security risks in certain scenarios.             Please read RFC 4941 for details. Reference: RFCs 4291 and 4941.

        :return: new link local IPv6 `IPAddress` object based on this `EUI`             using the technique described in RFC 4291.
        """
    return self.ipv6(338288524927261089654018896841347694592)