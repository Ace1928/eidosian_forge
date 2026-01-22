from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.tree import CommonTreeAdaptor
from six.moves import range
import stringtemplate3
def getNodeST(self, adaptor, t):
    text = adaptor.getText(t)
    nodeST = self._nodeST.getInstanceOf()
    uniqueName = 'n%d' % self.getNodeNumber(t)
    nodeST.setAttribute('name', uniqueName)
    if text is not None:
        text = text.replace('"', '\\\\"')
    nodeST.setAttribute('text', text)
    return nodeST