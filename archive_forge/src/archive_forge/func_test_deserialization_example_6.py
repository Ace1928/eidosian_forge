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
def test_deserialization_example_6(self):
    """
        Test the deserialization of example 6 which is a simple entity
        description.
        """
    actual_doc = prov.ProvDocument.deserialize(source=os.path.join(DATA_PATH, 'example_06.xml'), format='xml')
    expected_document = prov.ProvDocument()
    ex_ns = expected_document.add_namespace(*EX_NS)
    expected_document.add_namespace(*EX_TR)
    expected_document.entity('tr:WD-prov-dm-20111215', ((prov.PROV_TYPE, ex_ns['Document']), ('ex:version', '2')))
    self.assertEqual(actual_doc, expected_document)