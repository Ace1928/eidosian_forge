import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_stack_by_id_wait(self):
    marker_event = fakes.make_fake_stack_event(self.stack_id, self.stack_name, status='CREATE_COMPLETE', resource_name='name')
    marker_qs = 'marker={e_id}&sort_dir=asc'.format(e_id=marker_event['id'])
    resolve = 'resolve_outputs=False'
    self.register_uris([dict(method='GET', uri='{endpoint}/stacks/{id}?{resolve}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, resolve=resolve), status_code=302, headers=dict(location='{endpoint}/stacks/{name}/{id}?{resolve}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name, resolve=resolve))), dict(method='GET', uri='{endpoint}/stacks/{name}/{id}?{resolve}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name, resolve=resolve), json={'stack': self.stack}), dict(method='GET', uri='{endpoint}/stacks/{id}/events?{qs}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, qs='limit=1&sort_dir=desc'), complete_qs=True, json={'events': [marker_event]}), dict(method='DELETE', uri='{endpoint}/stacks/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id)), dict(method='GET', uri='{endpoint}/stacks/{id}/events?{qs}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, qs=marker_qs), complete_qs=True, json={'events': [fakes.make_fake_stack_event(self.stack_id, self.stack_name, status='DELETE_COMPLETE')]}), dict(method='GET', uri='{endpoint}/stacks/{id}?{resolve}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, resolve=resolve), status_code=404)])
    self.assertTrue(self.cloud.delete_stack(self.stack_id, wait=True))
    self.assert_calls()