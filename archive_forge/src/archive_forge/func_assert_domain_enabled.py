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
def assert_domain_enabled(self, domain_id, domain=None):
    """Assert the Domain is enabled.

        :raise AssertionError: if domain is disabled.
        """
    if domain is None:
        domain = self.get_domain(domain_id)
    if not domain.get('enabled', True):
        raise AssertionError(_('Domain is disabled: %s') % domain_id)