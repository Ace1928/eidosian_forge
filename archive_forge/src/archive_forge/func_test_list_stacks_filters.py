import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_stacks_filters(self):
    fake_stacks = [self.stack, fakes.make_fake_stack(self.getUniqueString('id'), self.getUniqueString('name'))]
    self.register_uris([dict(method='GET', uri=self.get_mock_url('orchestration', 'public', append=['stacks'], qs_elements=['name=a', 'status=b']), json={'stacks': fake_stacks})])
    stacks = self.cloud.list_stacks(name='a', status='b')
    [self._compare_stacks(b, a) for a, b in zip(stacks, fake_stacks)]
    self.assert_calls()