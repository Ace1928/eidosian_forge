from netaddr.ip import IPNetwork, cidr_exclude, cidr_merge
def remove_subnet(self, ip_network):
    """Remove a specified IPNetwork from available address space."""
    self._subnets.remove(ip_network)