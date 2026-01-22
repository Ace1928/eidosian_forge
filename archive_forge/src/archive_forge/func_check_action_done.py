import copy
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.senlin import res_base
from heat.engine import translation
def check_action_done(self, bindings):
    ret = True
    if not bindings:
        return ret
    for bd in bindings:
        if bd.get('finished', False):
            continue
        action = self.client().get_action(bd['action'])
        if action.status == self.ACTION_SUCCEEDED:
            bd['finished'] = True
        elif action.status == self.ACTION_FAILED:
            err_msg = _('Failed to execute %(action)s for %(cluster)s: %(reason)s') % {'action': action.action, 'cluster': bd[self.BD_CLUSTER], 'reason': action.status_reason}
            raise exception.ResourceInError(status_reason=err_msg, resource_status=self.FAILED)
        else:
            ret = False
    return ret