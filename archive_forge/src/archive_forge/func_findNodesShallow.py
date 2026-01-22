import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def findNodesShallow(parent, matcher, accum=None):
    if accum is None:
        accum = []
    if not parent.hasChildNodes():
        return accum
    for child in parent.childNodes:
        if matcher(child):
            accum.append(child)
        else:
            findNodes(child, matcher, accum)
    return accum