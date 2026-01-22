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
def stack_events(self, stack, resource_name=None, **attr):
    """Get a stack events

        :param stack: The value can be the ID of a stack or an instance of
            :class:`~openstack.orchestration.v1.stack.Stack`
        :param resource_name: The name of resource. If the resource_name is not None,
            the base_path changes.

        :returns: A generator of stack_events objects
        :rtype: :class:`~openstack.orchestration.v1.stack_event.StackEvent`
        """
    if isinstance(stack, _stack.Stack):
        obj = stack
    else:
        obj = self._get(_stack.Stack, stack)
    if resource_name:
        base_path = '/stacks/%(stack_name)s/%(stack_id)s/resources/%(resource_name)s/events'
        return self._list(_stack_event.StackEvent, stack_name=obj.name, stack_id=obj.id, resource_name=resource_name, base_path=base_path, **attr)
    return self._list(_stack_event.StackEvent, stack_name=obj.name, stack_id=obj.id, **attr)