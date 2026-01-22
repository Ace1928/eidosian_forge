from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.tree import CommonTreeAdaptor
from six.moves import range
import stringtemplate3
def toDOTDefineEdges(self, tree, adaptor, treeST, edgeST):
    if tree is None:
        return
    n = adaptor.getChildCount(tree)
    if n == 0:
        return
    parentName = 'n%d' % self.getNodeNumber(tree)
    parentText = adaptor.getText(tree)
    for i in range(n):
        child = adaptor.getChild(tree, i)
        childText = adaptor.getText(child)
        childName = 'n%d' % self.getNodeNumber(child)
        edgeST = edgeST.getInstanceOf()
        edgeST.setAttribute('parent', parentName)
        edgeST.setAttribute('child', childName)
        edgeST.setAttribute('parentText', parentText)
        edgeST.setAttribute('childText', childText)
        treeST.setAttribute('edges', edgeST)
        self.toDOTDefineEdges(child, adaptor, treeST, edgeST)