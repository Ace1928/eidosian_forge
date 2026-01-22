import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_search_stacks(self):
    fake_stacks = [self.stack, fakes.make_fake_stack(self.getUniqueString('id'), self.getUniqueString('name'))]
    self.register_uris([dict(method='GET', uri='{endpoint}/stacks'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT), json={'stacks': fake_stacks})])
    stacks = self.cloud.search_stacks()
    [self._compare_stacks(b, a) for a, b in zip(stacks, fake_stacks)]
    self.assert_calls()