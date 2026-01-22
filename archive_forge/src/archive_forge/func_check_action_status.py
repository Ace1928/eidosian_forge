from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def check_action_status(self, action_id):
    action = self.client().get_action(action_id)
    if action.status == 'SUCCEEDED':
        return True
    elif action.status == 'FAILED':
        raise exception.ResourceInError(status_reason=action.status_reason, resource_status=action.status)
    return False