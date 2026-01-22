from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.tree import CommonTreeAdaptor
from six.moves import range
import stringtemplate3
def toDOTDefineNodes(self, tree, adaptor, treeST, knownNodes=None):
    if knownNodes is None:
        knownNodes = set()
    if tree is None:
        return
    n = adaptor.getChildCount(tree)
    if n == 0:
        return
    number = self.getNodeNumber(tree)
    if number not in knownNodes:
        parentNodeST = self.getNodeST(adaptor, tree)
        treeST.setAttribute('nodes', parentNodeST)
        knownNodes.add(number)
    for i in range(n):
        child = adaptor.getChild(tree, i)
        number = self.getNodeNumber(child)
        if number not in knownNodes:
            nodeST = self.getNodeST(adaptor, child)
            treeST.setAttribute('nodes', nodeST)
            knownNodes.add(number)
        self.toDOTDefineNodes(child, adaptor, treeST, knownNodes)