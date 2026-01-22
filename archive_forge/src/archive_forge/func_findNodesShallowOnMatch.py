import warnings
from io import StringIO
from incremental import Version, getVersionString
from twisted.web import microdom
from twisted.web.microdom import escape, getElementsByTagName, unescape
def findNodesShallowOnMatch(parent, matcher, recurseMatcher, accum=None):
    if accum is None:
        accum = []
    if not parent.hasChildNodes():
        return accum
    for child in parent.childNodes:
        if matcher(child):
            accum.append(child)
        if recurseMatcher(child):
            findNodesShallowOnMatch(child, matcher, recurseMatcher, accum)
    return accum