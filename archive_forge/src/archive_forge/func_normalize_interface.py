import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
@staticmethod
def normalize_interface(interface):
    if interface:
        interface = interface.rstrip('URL')
    return interface