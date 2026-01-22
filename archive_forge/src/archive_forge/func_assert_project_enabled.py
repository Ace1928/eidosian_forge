from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def assert_project_enabled(self, project_id, project=None):
    """Assert the project is enabled and its associated domain is enabled.

        :raise AssertionError: if the project or domain is disabled.
        """
    if project is None:
        project = self.get_project(project_id)
    if project['domain_id']:
        self.assert_domain_enabled(domain_id=project['domain_id'])
    if not project.get('enabled', True):
        raise AssertionError(_('Project is disabled: %s') % project_id)