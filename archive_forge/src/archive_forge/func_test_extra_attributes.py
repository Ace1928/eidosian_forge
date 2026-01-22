import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_extra_attributes(self):
    document = ProvDocument()
    inf = document.influence(EX_NS['a2'], EX_NS['a1'], identifier=EX_NS['inf7'])
    add_labels(inf)
    add_types(inf)
    add_further_attributes(inf)
    self.assertEqual(len(inf.attributes), len(list(inf.formal_attributes) + inf.extra_attributes))