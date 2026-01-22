import struct
import sys
def subnet(self, prefixlen_diff=1, new_prefix=None):
    """Return a list of subnets, rather than an iterator."""
    return list(self.iter_subnets(prefixlen_diff, new_prefix))