from oslotest import base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
def test_register_funcs(self):
    resources = ['A', 'B', 'C']
    for r in resources:
        resource_extend.register_funcs(r, (lambda x: x,))
    for r in resources:
        self.assertIsNotNone(resource_extend.get_funcs(r))