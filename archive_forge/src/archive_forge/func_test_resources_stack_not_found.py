from unittest import mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.orchestration.v1 import _proxy
from openstack.orchestration.v1 import resource
from openstack.orchestration.v1 import software_config as sc
from openstack.orchestration.v1 import software_deployment as sd
from openstack.orchestration.v1 import stack
from openstack.orchestration.v1 import stack_environment
from openstack.orchestration.v1 import stack_event
from openstack.orchestration.v1 import stack_files
from openstack.orchestration.v1 import stack_template
from openstack.orchestration.v1 import template
from openstack import proxy
from openstack.tests.unit import test_proxy_base
@mock.patch.object(stack.Stack, 'find')
@mock.patch.object(resource.Resource, 'list')
def test_resources_stack_not_found(self, mock_list, mock_find):
    stack_name = 'test_stack'
    mock_find.side_effect = exceptions.ResourceNotFound('No stack found for test_stack')
    ex = self.assertRaises(exceptions.ResourceNotFound, self.proxy.resources, stack_name)
    self.assertEqual('No stack found for test_stack', str(ex))