import difflib
import glob
import inspect
import io
from lxml import etree
import os
import unittest
import warnings
from prov.identifier import Namespace, QualifiedName
from prov.constants import PROV
import prov.model as prov
from prov.tests.test_model import AllTestsBase
from prov.tests.utility import RoundTripTestCase
def test_other_elements(self):
    """
        PROV XML uses the <prov:other> element to enable the storage of non
        PROV information in a PROV XML document. It will be ignored by this
        library a warning will be raised informing the user.
        """
    xml_string = '\n        <prov:document\n            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n            xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n            xmlns:prov="http://www.w3.org/ns/prov#"\n            xmlns:ex="http://example.com/ns/ex#">\n\n          <!-- prov statements go here -->\n\n          <prov:other>\n            <ex:foo>\n              <ex:content>bar</ex:content>\n            </ex:foo>\n          </prov:other>\n\n          <!-- more prov statements can go here -->\n\n        </prov:document>\n        '
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with io.StringIO() as xml:
            xml.write(xml_string)
            xml.seek(0, 0)
            doc = prov.ProvDocument.deserialize(source=xml, format='xml')
    self.assertEqual(len(w), 1)
    self.assertTrue('Document contains non-PROV information in <prov:other>. It will be ignored in this package.' in str(w[0].message))
    self.assertEqual(len(doc._records), 0)