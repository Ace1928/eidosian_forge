import copy
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from oslo_context import context
from glance.api import policy
def elevated(self):
    """Return a copy of this context with admin flag set."""
    context = copy.copy(self)
    context.roles = copy.deepcopy(self.roles)
    if 'admin' not in context.roles:
        context.roles.append('admin')
    context.is_admin = True
    return context