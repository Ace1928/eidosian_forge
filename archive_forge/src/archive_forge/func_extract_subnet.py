from netaddr.ip import IPNetwork, cidr_exclude, cidr_merge
def extract_subnet(self, prefix, count=None):
    """Extract 1 or more subnets of size specified by CIDR prefix."""
    for cidr in self.available_subnets():
        subnets = list(cidr.subnet(prefix, count=count))
        if not subnets:
            continue
        self.remove_subnet(cidr)
        self._subnets = self._subnets.union(set(cidr_exclude(cidr, cidr_merge(subnets)[0])))
        return subnets
    return []