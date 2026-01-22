import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_serialize_to_path(self):
    document = ProvDocument()
    document.serialize('output.json')
    os.remove('output.json')
    document.serialize('http://netloc/outputmyprov/submit.php')