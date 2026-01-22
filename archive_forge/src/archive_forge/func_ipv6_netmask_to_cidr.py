from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ipaddress import ip_interface, ip_network
def ipv6_netmask_to_cidr(mask):
    """converts an IPv6 netmask to CIDR form

    According to the link below, CIDR is the only official way to specify
    a subset of IPv6. With that said, the same link provides a way to
    loosely convert an netmask to a CIDR.

    Arguments:
      mask (string): The IPv6 netmask to convert to CIDR

    Returns:
      int: The CIDR representation of the netmask

    References:
      https://stackoverflow.com/a/33533007
      http://v6decode.com/
    """
    bit_masks = [0, 32768, 49152, 57344, 61440, 63488, 64512, 65024, 65280, 65408, 65472, 65504, 65520, 65528, 65532, 65534, 65535]
    count = 0
    try:
        for w in mask.split(':'):
            if not w or int(w, 16) == 0:
                break
            count += bit_masks.index(int(w, 16))
        return count
    except Exception:
        return -1