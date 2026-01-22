from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def ovirtvmip(self, ovirt_vms, attr=None, network_ip=None):
    """Return first IP"""
    return self.__get_first_ip(self.ovirtvmips(ovirt_vms, attr))