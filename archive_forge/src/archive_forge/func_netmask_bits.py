import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def netmask_bits(self):
    """
        @return: If this IP is a valid netmask, the number of non-zero
            bits are returned, otherwise it returns the width in bits for
            the IP address version.
        """
    if not self.is_netmask():
        return self._module.width
    if self._value == 0:
        return 0
    i_val = self._value
    numbits = 0
    while i_val > 0:
        if i_val & 1 == 1:
            break
        numbits += 1
        i_val >>= 1
    mask_length = self._module.width - numbits
    if not 0 <= mask_length <= self._module.width:
        raise ValueError('Unexpected mask length %d for address type!' % mask_length)
    return mask_length