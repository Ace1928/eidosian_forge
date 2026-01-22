import testtools
from heatclient.common import base
from heatclient.v1 import events
from heatclient.v1 import stacks
def test_is_diff_object_with_no_id(self):
    r1 = base.Resource(None, {'name': 'joe', 'age': 12})
    r2 = base.Resource(None, {'name': 'joe', 'age': 12})
    self.assertFalse(r1.is_same_obj(r2))
    self.assertFalse(r2.is_same_obj(r1))