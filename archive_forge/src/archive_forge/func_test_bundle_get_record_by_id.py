import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_bundle_get_record_by_id(self):
    document = ProvDocument()
    self.assertEqual(document.get_record(None), None)