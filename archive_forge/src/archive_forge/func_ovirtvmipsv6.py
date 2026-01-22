from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def ovirtvmipsv6(self, ovirt_vms, attr=None, network_ip=None):
    """Return list of IPv6 IPs"""
    return self._parse_ips(ovirt_vms, lambda version: version == 'v6', attr)