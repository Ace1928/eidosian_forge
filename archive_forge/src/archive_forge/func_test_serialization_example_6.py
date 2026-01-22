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
def test_serialization_example_6(self):
    """
        Test the serialization of example 6 which is a simple entity
        description.
        """
    document = prov.ProvDocument()
    ex_ns = document.add_namespace(*EX_NS)
    document.add_namespace(*EX_TR)
    document.entity('tr:WD-prov-dm-20111215', ((prov.PROV_TYPE, ex_ns['Document']), ('ex:version', '2')))
    with io.BytesIO() as actual:
        document.serialize(format='xml', destination=actual)
        compare_xml(os.path.join(DATA_PATH, 'example_06.xml'), actual)