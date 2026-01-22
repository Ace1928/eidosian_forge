from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
def getDocument(self):
    return self.document._elementTree