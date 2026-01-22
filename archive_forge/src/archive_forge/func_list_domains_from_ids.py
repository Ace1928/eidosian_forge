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
def list_domains_from_ids(self, domain_ids):
    """List domains for the provided list of ids.

        :param domain_ids: list of ids

        :returns: a list of domain_refs.

        This method is used internally by the assignment manager to bulk read
        a set of domains given their ids.

        """
    projects = self.list_projects_from_ids(domain_ids)
    domains = [self._get_domain_from_project(project) for project in projects]
    return domains