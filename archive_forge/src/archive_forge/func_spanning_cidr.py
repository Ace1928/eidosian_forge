import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def spanning_cidr(ip_addrs):
    """
    Function that accepts a sequence of IP addresses and subnets returning
    a single `IPNetwork` subnet that is large enough to span the lower and
    upper bound IP addresses with a possible overlap on either end.

    :param ip_addrs: sequence of IP addresses and subnets.

    :return: a single spanning `IPNetwork` subnet.
    """
    ip_addrs_iter = iter(ip_addrs)
    try:
        network_a = IPNetwork(next(ip_addrs_iter))
        network_b = IPNetwork(next(ip_addrs_iter))
    except StopIteration:
        raise ValueError('IP sequence must contain at least 2 elements!')
    if network_a < network_b:
        min_network = network_a
        max_network = network_b
    else:
        min_network = network_b
        max_network = network_a
    for ip in ip_addrs_iter:
        network = IPNetwork(ip)
        if network < min_network:
            min_network = network
        if network > max_network:
            max_network = network
    if min_network.version != max_network.version:
        raise TypeError('IP sequence cannot contain both IPv4 and IPv6!')
    ipnum = max_network.last
    prefixlen = max_network.prefixlen
    lowest_ipnum = min_network.first
    width = max_network._module.width
    while prefixlen > 0 and ipnum > lowest_ipnum:
        prefixlen -= 1
        ipnum &= -(1 << width - prefixlen)
    return IPNetwork((ipnum, prefixlen), version=min_network.version)