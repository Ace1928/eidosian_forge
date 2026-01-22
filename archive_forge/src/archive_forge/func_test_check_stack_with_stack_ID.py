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
@mock.patch.object(stack.Stack, 'existing')
def test_check_stack_with_stack_ID(self, mock_stack):
    stk = mock.Mock()
    mock_stack.return_value = stk
    res = self.proxy.check_stack('FAKE_ID')
    self.assertIsNone(res)
    mock_stack.assert_called_once_with(id='FAKE_ID')
    stk.check.assert_called_once_with(self.proxy)