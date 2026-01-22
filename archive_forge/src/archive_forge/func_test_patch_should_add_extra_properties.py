import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_add_extra_properties(self):
    obj = {'name': 'fred'}
    original = self.model(obj)
    original['weight'] = '10'
    patch = original.patch
    expected = '[{"path": "/weight", "value": "10", "op": "add"}]'
    self.assertTrue(compare_json_patches(patch, expected))