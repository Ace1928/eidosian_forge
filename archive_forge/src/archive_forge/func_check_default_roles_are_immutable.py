from oslo_policy import _checks
from oslo_policy import policy
from oslo_upgradecheck import common_checks
from oslo_upgradecheck import upgradecheck
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import rbac_enforcer
import keystone.conf
from keystone.server import backends
def check_default_roles_are_immutable(self):
    hints = driver_hints.Hints()
    hints.add_filter('domain_id', None)
    roles = PROVIDERS.role_api.list_roles(hints=hints)
    default_roles = ('admin', 'member', 'reader')
    failed_roles = []
    for role in [r for r in roles if r['name'] in default_roles]:
        if not role.get('options', {}).get('immutable'):
            failed_roles.append(role['name'])
    if any(failed_roles):
        return upgradecheck.Result(upgradecheck.Code.FAILURE, 'Roles are not immutable: %s' % ', '.join(failed_roles))
    return upgradecheck.Result(upgradecheck.Code.SUCCESS, 'Default roles are immutable.')