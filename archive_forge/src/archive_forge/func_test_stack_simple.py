import tempfile
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.functional import base
def test_stack_simple(self):
    test_template = tempfile.NamedTemporaryFile(delete=False)
    test_template.write(fakes.FAKE_TEMPLATE.encode('utf-8'))
    test_template.close()
    self.stack_name = self.getUniqueString('simple_stack')
    self.addCleanup(self._cleanup_stack)
    stack = self.user_cloud.create_stack(name=self.stack_name, template_file=test_template.name, wait=True)
    self.assertEqual('CREATE_COMPLETE', stack['stack_status'])
    rand = stack['outputs'][0]['output_value']
    self.assertEqual(10, len(rand))
    stack = self.user_cloud.get_stack(self.stack_name)
    self.assertEqual('CREATE_COMPLETE', stack['stack_status'])
    self.assertEqual(rand, stack['outputs'][0]['output_value'])
    stacks = self.user_cloud.list_stacks()
    stack_ids = [s['id'] for s in stacks]
    self.assertIn(stack['id'], stack_ids)
    stack = self.user_cloud.update_stack(self.stack_name, template_file=test_template.name, wait=True)
    self.assertEqual('UPDATE_COMPLETE', stack['stack_status'])
    rand = stack['outputs'][0]['output_value']
    self.assertEqual(rand, stack['outputs'][0]['output_value'])
    stack = self.user_cloud.update_stack(self.stack_name, template_file=test_template.name, wait=True, length=12)
    stack = self.user_cloud.get_stack(self.stack_name)
    self.assertEqual('UPDATE_COMPLETE', stack['stack_status'])
    new_rand = stack['outputs'][0]['output_value']
    self.assertNotEqual(rand, new_rand)
    self.assertEqual(12, len(new_rand))