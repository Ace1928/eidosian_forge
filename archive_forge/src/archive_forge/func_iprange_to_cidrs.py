import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def iprange_to_cidrs(start, end):
    """
    A function that accepts an arbitrary start and end IP address or subnet
    and returns a list of CIDR subnets that fit exactly between the boundaries
    of the two with no overlap.

    :param start: the start IP address or subnet.

    :param end: the end IP address or subnet.

    :return: a list of one or more IP addresses and subnets.
    """
    cidr_list = []
    start = IPNetwork(start)
    end = IPNetwork(end)
    iprange = [start.first, end.last]
    cidr_span = spanning_cidr([start, end])
    width = start._module.width
    if cidr_span.first < iprange[0]:
        exclude = IPNetwork((iprange[0] - 1, width), version=start.version)
        cidr_list = cidr_partition(cidr_span, exclude)[2]
        cidr_span = cidr_list.pop()
    if cidr_span.last > iprange[1]:
        exclude = IPNetwork((iprange[1] + 1, width), version=start.version)
        cidr_list += cidr_partition(cidr_span, exclude)[0]
    else:
        cidr_list.append(cidr_span)
    return cidr_list