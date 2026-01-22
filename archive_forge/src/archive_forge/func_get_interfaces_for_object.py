import importlib
from threading import Lock
import gi
from ._gi import \
from .types import \
from ._constants import \
def get_interfaces_for_object(object_info):
    interfaces = []
    for interface_info in object_info.get_interfaces():
        namespace = interface_info.get_namespace()
        name = interface_info.get_name()
        module = importlib.import_module('gi.repository.' + namespace)
        interfaces.append(getattr(module, name))
    return interfaces