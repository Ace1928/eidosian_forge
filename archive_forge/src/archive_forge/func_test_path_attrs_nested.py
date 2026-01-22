from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_path_attrs_nested(self):
    files = {'randomstr.yaml': self.template_randomstr}
    stack_id = self.stack_create(template=self.template_nested, files=files)
    expected_resources = {'random_group': 'OS::Heat::AutoScalingGroup'}
    self.assertEqual(expected_resources, self.list_resources(stack_id))
    self._assert_output_values(stack_id)