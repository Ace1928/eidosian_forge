from netaddr.ip import IPNetwork, cidr_exclude, cidr_merge
def available_subnets(self):
    """Returns a list of the currently available subnets."""
    return sorted(self._subnets, key=lambda x: x.prefixlen, reverse=True)