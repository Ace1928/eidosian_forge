from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def ovirtvmipv4(self, ovirt_vms, attr=None, network_ip=None):
    """Return first IPv4 IP"""
    return self.__get_first_ip(self.ovirtvmipsv4(ovirt_vms, attr, network_ip))