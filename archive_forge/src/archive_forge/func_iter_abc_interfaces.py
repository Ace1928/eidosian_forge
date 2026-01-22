import unittest
from zope.interface.common import ABCInterface
from zope.interface.common import ABCInterfaceClass
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
def iter_abc_interfaces(predicate=lambda iface: True):
    seen = set()
    stack = list(ABCInterface.dependents)
    while stack:
        iface = stack.pop(0)
        if iface in seen or not isinstance(iface, ABCInterfaceClass):
            continue
        seen.add(iface)
        stack.extend(list(iface.dependents))
        if not predicate(iface):
            continue
        registered = set(iface.getRegisteredConformers())
        registered -= set(iface._ABCInterfaceClass__ignored_classes)
        if registered:
            yield (iface, registered)