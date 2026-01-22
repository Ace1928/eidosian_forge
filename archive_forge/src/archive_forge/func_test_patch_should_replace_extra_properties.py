import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_replace_extra_properties(self):
    obj = {'name': 'fred', 'weight': '10'}
    original = self.model(obj)
    original['weight'] = '22'
    patch = original.patch
    expected = '[{"path": "/weight", "value": "22", "op": "replace"}]'
    self.assertTrue(compare_json_patches(patch, expected))