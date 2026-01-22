import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def get_mac_addr_by_ipv6(ipv6, dialect=netaddr.mac_unix_expanded):
    """Extract MAC address from interface identifier based IPv6 address.

    For example from link-local addresses (fe80::/10) generated from MAC.

    :param ipv6: An interface identifier (i.e. mostly MAC) based IPv6
                 address as a netaddr.IPAddress() object.
    :param dialect: The netaddr dialect of the the object returned.
                    Defaults to netaddr.mac_unix_expanded.
    :returns: A MAC address as a netaddr.EUI() object.

    See also:
    * https://tools.ietf.org/html/rfc4291#appendix-A
    * https://tools.ietf.org/html/rfc4291#section-2.5.6

    .. versionadded:: 4.3.0
    """
    return netaddr.EUI(int(((ipv6 & 18446742974197923840) >> 16) + (ipv6 & 16777215) ^ 2199023255552), dialect=dialect)