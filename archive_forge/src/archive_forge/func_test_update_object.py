import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_object(self):
    properties = {'description': 'UPDATED_DESCRIPTION'}
    obj = self.controller.update(NAMESPACE1, OBJECT1, **properties)
    self.assertEqual(OBJECT1, obj.name)