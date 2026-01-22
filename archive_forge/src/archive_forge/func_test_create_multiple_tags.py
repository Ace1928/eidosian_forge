import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_create_multiple_tags(self):
    properties = {'tags': [TAGNEW2, TAGNEW3]}
    tags = self.controller.create_multiple(NAMESPACE1, **properties)
    self.assertEqual([TAGNEW2, TAGNEW3], tags)