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
def test_serialization_example_7(self):
    """
        Test the serialization of example 7 which is a basic activity.
        """
    document = prov.ProvDocument()
    document.add_namespace(*EX_NS)
    document.activity('ex:a1', '2011-11-16T16:05:00', '2011-11-16T16:06:00', [(prov.PROV_TYPE, prov.Literal('ex:edit', prov.XSD_QNAME)), ('ex:host', 'server.example.org')])
    with io.BytesIO() as actual:
        document.serialize(format='xml', destination=actual)
        compare_xml(os.path.join(DATA_PATH, 'example_07.xml'), actual)