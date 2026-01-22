from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_findElementsWithAttribute(self):
    doc1 = self.dom.parseString('<a foo="1"><b foo="2"/><c foo="1"/><d/></a>')
    node_list = domhelpers.findElementsWithAttribute(doc1, 'foo')
    actual = ''.join([node.tagName for node in node_list])
    self.assertEqual(actual, 'abc')
    node_list = domhelpers.findElementsWithAttribute(doc1, 'foo', '1')
    actual = ''.join([node.tagName for node in node_list])
    self.assertEqual(actual, 'ac')