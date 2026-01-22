from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def processStartTag(self, token):
    currentNode = self.tree.openElements[-1]
    if token['name'] in self.breakoutElements or (token['name'] == 'font' and set(token['data'].keys()) & {'color', 'face', 'size'}):
        self.parser.parseError('unexpected-html-element-in-foreign-content', {'name': token['name']})
        while self.tree.openElements[-1].namespace != self.tree.defaultNamespace and (not self.parser.isHTMLIntegrationPoint(self.tree.openElements[-1])) and (not self.parser.isMathMLTextIntegrationPoint(self.tree.openElements[-1])):
            self.tree.openElements.pop()
        return token
    else:
        if currentNode.namespace == namespaces['mathml']:
            self.parser.adjustMathMLAttributes(token)
        elif currentNode.namespace == namespaces['svg']:
            self.adjustSVGTagNames(token)
            self.parser.adjustSVGAttributes(token)
        self.parser.adjustForeignAttributes(token)
        token['namespace'] = currentNode.namespace
        self.tree.insertElement(token)
        if token['selfClosing']:
            self.tree.openElements.pop()
            token['selfClosingAcknowledged'] = True