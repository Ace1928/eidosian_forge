from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
def visitFunctionDef(self, node):
    if self.classname:
        entity = '%s%s' % (self.classname, node.name)
    else:
        entity = node.name
    name = '%d:%d: %r' % (node.lineno, node.col_offset, entity)
    if self.graph is not None:
        pathnode = self.appendPathNode(name)
        self.tail = pathnode
        self.dispatch_list(node.body)
        bottom = PathNode('', look='point')
        self.graph.connect(self.tail, bottom)
        self.graph.connect(pathnode, bottom)
        self.tail = bottom
    else:
        self.graph = PathGraph(name, entity, node.lineno, node.col_offset)
        pathnode = PathNode(name)
        self.tail = pathnode
        self.dispatch_list(node.body)
        self.graphs['%s%s' % (self.classname, node.name)] = self.graph
        self.reset()