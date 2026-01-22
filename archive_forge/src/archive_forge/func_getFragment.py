from html5lib.treebuilders import _base, etree as etree_builders
from lxml import html, etree
def getFragment(self):
    fragment = []
    element = self.openElements[0]._element
    if element.text:
        fragment.append(element.text)
    fragment.extend(element.getchildren())
    if element.tail:
        fragment.append(element.tail)
    return fragment