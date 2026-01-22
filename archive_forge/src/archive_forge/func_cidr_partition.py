import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def cidr_partition(target, exclude):
    """
    Partitions a target IP subnet on an exclude IP address.

    :param target: the target IP address or subnet to be divided up.

    :param exclude: the IP address or subnet to partition on

    :return: list of `IPNetwork` objects before, the partition and after, sorted.

    Adding the three lists returns the equivalent of the original subnet.
    """
    target = IPNetwork(target)
    exclude = IPNetwork(exclude)
    if exclude.last < target.first:
        return ([], [], [target.cidr])
    elif target.last < exclude.first:
        return ([target.cidr], [], [])
    if target.prefixlen >= exclude.prefixlen:
        return ([], [target], [])
    left = []
    right = []
    new_prefixlen = target.prefixlen + 1
    target_module_width = target._module.width
    target_first = target.first
    version = exclude.version
    i_lower = target_first
    i_upper = target_first + 2 ** (target_module_width - new_prefixlen)
    while exclude.prefixlen >= new_prefixlen:
        if exclude.first >= i_upper:
            left.append(IPNetwork((i_lower, new_prefixlen), version=version))
            matched = i_upper
        else:
            right.append(IPNetwork((i_upper, new_prefixlen), version=version))
            matched = i_lower
        new_prefixlen += 1
        if new_prefixlen > target_module_width:
            break
        i_lower = matched
        i_upper = matched + 2 ** (target_module_width - new_prefixlen)
    return (left, [exclude], right[::-1])