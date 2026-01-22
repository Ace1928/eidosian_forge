import itertools as _itertools
import sys as _sys
from netaddr.ip import IPNetwork, IPAddress, IPRange, cidr_merge, cidr_exclude, iprange_to_cidrs
def iter_cidrs(self):
    """
        :return: an iterator over individual IP subnets within this IP set.
        """
    return sorted(self._cidrs)