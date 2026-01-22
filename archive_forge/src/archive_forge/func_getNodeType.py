from collections import OrderedDict
from .Node import Node
def getNodeType(self, name):
    try:
        return self.nodeList[name]
    except KeyError:
        raise Exception("No node type called '%s'" % name)