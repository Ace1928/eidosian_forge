import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def findElementsWithAttribute(parent, attribute, value=None):
    if value:
        return findElements(parent, lambda n, attribute=attribute, value=value: n.hasAttribute(attribute) and n.getAttribute(attribute) == value)
    else:
        return findElements(parent, lambda n, attribute=attribute: n.hasAttribute(attribute))