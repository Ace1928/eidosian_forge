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
def get_stack_template(self, stack):
    """Get template used by a stack

        :param stack: The value can be the ID of a stack or an instance of
            :class:`~openstack.orchestration.v1.stack.Stack`

        :returns: One object of
            :class:`~openstack.orchestration.v1.stack_template.StackTemplate`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    if isinstance(stack, _stack.Stack):
        obj = stack
    else:
        obj = self._find(_stack.Stack, stack, ignore_missing=False)
    return self._get(_stack_template.StackTemplate, requires_id=False, stack_name=obj.name, stack_id=obj.id)