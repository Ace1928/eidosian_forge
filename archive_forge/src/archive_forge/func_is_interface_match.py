import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def is_interface_match(self, endpoint, interface):
    try:
        return interface == endpoint['interface']
    except KeyError:
        return False