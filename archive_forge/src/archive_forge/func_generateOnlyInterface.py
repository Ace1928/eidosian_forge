from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def generateOnlyInterface(list, int):
    """Filters items in a list by class"""
    for n in list:
        if int.providedBy(n):
            yield n