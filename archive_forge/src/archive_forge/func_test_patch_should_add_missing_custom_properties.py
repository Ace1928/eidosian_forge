import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_add_missing_custom_properties(self):
    obj = {'name': 'fred'}
    original = self.model(obj)
    original['shape'] = 'circle'
    patch = original.patch
    expected = '[{"path": "/shape", "value": "circle", "op": "add"}]'
    self.assertTrue(compare_json_patches(patch, expected))