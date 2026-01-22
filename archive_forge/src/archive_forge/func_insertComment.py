from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
def insertComment(self, data, parent=None):
    if not self.rootInserted:
        self.initialComments.append(data)
    else:
        _base.TreeBuilder.insertComment(self, data, parent)