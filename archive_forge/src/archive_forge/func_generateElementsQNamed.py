from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def generateElementsQNamed(list, name, uri):
    """Filters Element items in a list with matching name and URI."""
    for n in list:
        if IElement.providedBy(n) and n.name == name and (n.uri == uri):
            yield n