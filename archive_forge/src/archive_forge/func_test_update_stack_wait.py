import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_stack_wait(self):
    marker_event = fakes.make_fake_stack_event(self.stack_id, self.stack_name, status='CREATE_COMPLETE', resource_name='name')
    marker_qs = 'marker={e_id}&sort_dir=asc'.format(e_id=marker_event['id'])
    test_template = tempfile.NamedTemporaryFile(delete=False)
    test_template.write(fakes.FAKE_TEMPLATE.encode('utf-8'))
    test_template.close()
    self.register_uris([dict(method='GET', uri='{endpoint}/stacks/{name}/events?{qs}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name, qs='limit=1&sort_dir=desc'), json={'events': [marker_event]}), dict(method='PUT', uri='{endpoint}/stacks/{name}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name), validate=dict(json={'disable_rollback': False, 'parameters': {}, 'tags': self.stack_tag, 'template': fakes.FAKE_TEMPLATE_CONTENT, 'timeout_mins': 60}), json={}), dict(method='GET', uri='{endpoint}/stacks/{name}/events?{qs}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name, qs=marker_qs), json={'events': [fakes.make_fake_stack_event(self.stack_id, self.stack_name, status='UPDATE_COMPLETE', resource_name='name')]}), dict(method='GET', uri='{endpoint}/stacks/{name}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, name=self.stack_name), status_code=302, headers=dict(location='{endpoint}/stacks/{name}/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name))), dict(method='GET', uri='{endpoint}/stacks/{name}/{id}'.format(endpoint=fakes.ORCHESTRATION_ENDPOINT, id=self.stack_id, name=self.stack_name), json={'stack': self.stack})])
    self.cloud.update_stack(self.stack_name, tags=self.stack_tag, template_file=test_template.name, wait=True)
    self.assert_calls()