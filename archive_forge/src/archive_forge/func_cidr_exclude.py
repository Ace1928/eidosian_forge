import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def cidr_exclude(target, exclude):
    """
    Removes an exclude IP address or subnet from target IP subnet.

    :param target: the target IP address or subnet to be divided up.

    :param exclude: the IP address or subnet to be removed from target.

    :return: list of `IPNetwork` objects remaining after exclusion.
    """
    left, _, right = cidr_partition(target, exclude)
    return left + right