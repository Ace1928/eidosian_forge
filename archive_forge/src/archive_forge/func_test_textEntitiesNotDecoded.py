from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_textEntitiesNotDecoded(self):
    """
        Microdom does not decode entities in text nodes.
        """
    doc5_xml = '<x>Souffl&amp;</x>'
    doc5 = self.dom.parseString(doc5_xml)
    actual = domhelpers.gatherTextNodes(doc5)
    expected = 'Souffl&amp;'
    self.assertEqual(actual, expected)
    actual = domhelpers.gatherTextNodes(doc5.documentElement)
    self.assertEqual(actual, expected)