import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_replace_missing_core_properties(self):
    obj = {'name': 'fred'}
    original = self.model(obj)
    original['color'] = 'red'
    patch = original.patch
    expected = '[{"path": "/color", "value": "red", "op": "replace"}]'
    self.assertTrue(compare_json_patches(patch, expected))