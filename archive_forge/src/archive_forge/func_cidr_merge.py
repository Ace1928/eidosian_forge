import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def cidr_merge(ip_addrs):
    """
    A function that accepts an iterable sequence of IP addresses and subnets
    merging them into the smallest possible list of CIDRs. It merges adjacent
    subnets where possible, those contained within others and also removes
    any duplicates.

    :param ip_addrs: an iterable sequence of IP addresses, subnets or ranges.

    :return: a summarized list of `IPNetwork` objects.
    """
    if not hasattr(ip_addrs, '__iter__'):
        raise ValueError('A sequence or iterator is expected!')
    ranges = []
    for ip in ip_addrs:
        if isinstance(ip, (IPNetwork, IPRange)):
            net = ip
        else:
            net = IPNetwork(ip)
        ranges.append((net.version, net.last, net.first, net))
    ranges.sort()
    i = len(ranges) - 1
    while i > 0:
        if ranges[i][0] == ranges[i - 1][0] and ranges[i][2] - 1 <= ranges[i - 1][1]:
            ranges[i - 1] = (ranges[i][0], ranges[i][1], min(ranges[i - 1][2], ranges[i][2]))
            del ranges[i]
        i -= 1
    merged = []
    for range_tuple in ranges:
        if len(range_tuple) == 4:
            original = range_tuple[3]
            if isinstance(original, IPRange):
                merged.extend(original.cidrs())
            else:
                merged.append(original)
        else:
            version = range_tuple[0]
            range_start = IPAddress(range_tuple[2], version=version)
            range_stop = IPAddress(range_tuple[1], version=version)
            merged.extend(iprange_to_cidrs(range_start, range_stop))
    return merged