from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def endTagFormatting(self, token):
    """The much-feared adoption agency algorithm"""
    outerLoopCounter = 0
    while outerLoopCounter < 8:
        outerLoopCounter += 1
        formattingElement = self.tree.elementInActiveFormattingElements(token['name'])
        if not formattingElement or (formattingElement in self.tree.openElements and (not self.tree.elementInScope(formattingElement.name))):
            self.endTagOther(token)
            return
        elif formattingElement not in self.tree.openElements:
            self.parser.parseError('adoption-agency-1.2', {'name': token['name']})
            self.tree.activeFormattingElements.remove(formattingElement)
            return
        elif not self.tree.elementInScope(formattingElement.name):
            self.parser.parseError('adoption-agency-4.4', {'name': token['name']})
            return
        elif formattingElement != self.tree.openElements[-1]:
            self.parser.parseError('adoption-agency-1.3', {'name': token['name']})
        afeIndex = self.tree.openElements.index(formattingElement)
        furthestBlock = None
        for element in self.tree.openElements[afeIndex:]:
            if element.nameTuple in specialElements:
                furthestBlock = element
                break
        if furthestBlock is None:
            element = self.tree.openElements.pop()
            while element != formattingElement:
                element = self.tree.openElements.pop()
            self.tree.activeFormattingElements.remove(element)
            return
        commonAncestor = self.tree.openElements[afeIndex - 1]
        bookmark = self.tree.activeFormattingElements.index(formattingElement)
        lastNode = node = furthestBlock
        innerLoopCounter = 0
        index = self.tree.openElements.index(node)
        while innerLoopCounter < 3:
            innerLoopCounter += 1
            index -= 1
            node = self.tree.openElements[index]
            if node not in self.tree.activeFormattingElements:
                self.tree.openElements.remove(node)
                continue
            if node == formattingElement:
                break
            if lastNode == furthestBlock:
                bookmark = self.tree.activeFormattingElements.index(node) + 1
            clone = node.cloneNode()
            self.tree.activeFormattingElements[self.tree.activeFormattingElements.index(node)] = clone
            self.tree.openElements[self.tree.openElements.index(node)] = clone
            node = clone
            if lastNode.parent:
                lastNode.parent.removeChild(lastNode)
            node.appendChild(lastNode)
            lastNode = node
        if lastNode.parent:
            lastNode.parent.removeChild(lastNode)
        if commonAncestor.name in frozenset(('table', 'tbody', 'tfoot', 'thead', 'tr')):
            parent, insertBefore = self.tree.getTableMisnestedNodePosition()
            parent.insertBefore(lastNode, insertBefore)
        else:
            commonAncestor.appendChild(lastNode)
        clone = formattingElement.cloneNode()
        furthestBlock.reparentChildren(clone)
        furthestBlock.appendChild(clone)
        self.tree.activeFormattingElements.remove(formattingElement)
        self.tree.activeFormattingElements.insert(bookmark, clone)
        self.tree.openElements.remove(formattingElement)
        self.tree.openElements.insert(self.tree.openElements.index(furthestBlock) + 1, clone)