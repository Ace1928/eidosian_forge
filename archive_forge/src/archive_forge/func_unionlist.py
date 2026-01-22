from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def unionlist(*args):
    l = []
    for x in args:
        l.extend(x)
    d = {x: 1 for x in l}
    return d.keys()