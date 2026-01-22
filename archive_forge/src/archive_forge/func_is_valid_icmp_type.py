import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def is_valid_icmp_type(type):
    """Verify if ICMP type is valid.

    :param type: ICMP *type* field can only be a valid integer
    :returns: bool

    ICMP *type* field can be valid integer having a value of 0
    up to and including 255.
    """
    return _is_int_in_range(type, 0, 255)