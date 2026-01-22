from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def execute_actions(self, actions):
    all_executed = True
    for action in actions:
        if action['done']:
            continue
        all_executed = False
        if 'action_id' in action:
            if action['action_id'] is None:
                func = getattr(self.client(), action['func'])
                ret = func(**action['params'])
                if isinstance(ret, dict):
                    action['action_id'] = ret['action']
                else:
                    action['action_id'] = ret.location.split('/')[-1]
            else:
                ret = self.check_action_status(action['action_id'])
                action['done'] = ret
        else:
            ret = self.cluster_is_active(action['cluster_id'])
            action['done'] = ret
        break
    return all_executed