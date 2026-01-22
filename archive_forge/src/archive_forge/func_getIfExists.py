import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def getIfExists(node, nodeId):
    """
    Get a node with the specified C{nodeId} as any of the C{class},
    C{id} or C{pattern} attributes.  If there is no such node, return
    L{None}.
    """
    return _get(node, nodeId)