import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def findNodesNamed(parent, name):
    return findNodes(parent, lambda n, name=name: n.nodeName == name)