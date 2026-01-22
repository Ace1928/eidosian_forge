import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def iter_unique_ips(*args):
    """
    :param args: A list of IP addresses and subnets passed in as arguments.

    :return: A generator that flattens out IP subnets, yielding unique
        individual IP addresses (no duplicates).
    """
    for cidr in cidr_merge(args):
        for ip in cidr:
            yield ip