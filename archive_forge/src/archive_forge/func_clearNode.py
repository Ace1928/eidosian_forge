import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def clearNode(node):
    """
    Remove all children from the given node.
    """
    node.childNodes[:] = []