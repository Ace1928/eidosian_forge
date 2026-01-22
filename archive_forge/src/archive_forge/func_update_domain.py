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
def update_domain(self, domain_id, domain, initiator=None):
    self.assert_domain_not_federated(domain_id, domain)
    project = base.get_project_from_domain(domain)
    try:
        original_domain = self.driver.get_project(domain_id)
        project = self._update_project(domain_id, project, initiator)
    except exception.ProjectNotFound:
        raise exception.DomainNotFound(domain_id=domain_id)
    domain_from_project = self._get_domain_from_project(project)
    self.get_domain.invalidate(self, domain_id)
    self.get_domain_by_name.invalidate(self, original_domain['name'])
    return domain_from_project