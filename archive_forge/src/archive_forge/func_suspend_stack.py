from openstack import exceptions
from openstack.orchestration.util import template_utils
from openstack.orchestration.v1 import resource as _resource
from openstack.orchestration.v1 import software_config as _sc
from openstack.orchestration.v1 import software_deployment as _sd
from openstack.orchestration.v1 import stack as _stack
from openstack.orchestration.v1 import stack_environment as _stack_environment
from openstack.orchestration.v1 import stack_event as _stack_event
from openstack.orchestration.v1 import stack_files as _stack_files
from openstack.orchestration.v1 import stack_template as _stack_template
from openstack.orchestration.v1 import template as _template
from openstack import proxy
from openstack import resource
def suspend_stack(self, stack):
    """Suspend a stack status

        :param stack: The value can be either the ID of a stack or an instance
            of :class:`~openstack.orchestration.v1.stack.Stack`.
        :returns: ``None``
        """
    res = self._get_resource(_stack.Stack, stack)
    res.suspend(self)