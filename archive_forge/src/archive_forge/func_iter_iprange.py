import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def iter_iprange(start, end, step=1):
    """
    A generator that produces IPAddress objects between an arbitrary start
    and stop IP address with intervals of step between them. Sequences
    produce are inclusive of boundary IPs.

    :param start: start IP address.

    :param end: end IP address.

    :param step: (optional) size of step between IP addresses. Default: 1

    :return: an iterator of one or more `IPAddress` objects.
    """
    start = IPAddress(start)
    end = IPAddress(end)
    if start.version != end.version:
        raise TypeError('start and stop IP versions do not match!')
    version = start.version
    step = int(step)
    if step == 0:
        raise ValueError('step argument cannot be zero')
    start = int(start)
    stop = int(end)
    negative_step = False
    if step < 0:
        negative_step = True
    index = start - step
    while True:
        index += step
        if negative_step:
            if not index >= stop:
                break
        elif not index <= stop:
            break
        yield IPAddress(index, version)