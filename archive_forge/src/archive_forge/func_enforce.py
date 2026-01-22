from oslo_log import log
from keystone.common.rbac_enforcer import policy
from keystone import exception
from keystone.policy.backends import base
def enforce(self, credentials, action, target):
    msg = 'enforce %(action)s: %(credentials)s'
    LOG.debug(msg, {'action': action, 'credentials': credentials})
    policy.enforce(credentials, action, target)