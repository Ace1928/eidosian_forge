import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_create_namespace_invalid_data(self):
    properties = {}
    self.assertRaises(TypeError, self.controller.create, **properties)