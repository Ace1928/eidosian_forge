import uuid
from oslo_log import log
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from keystone.server import backends
def project_setup(self):
    try:
        project_id = self.project_id
        if project_id is None:
            project_id = uuid.uuid4().hex
        project = {'enabled': True, 'id': project_id, 'domain_id': self.default_domain_id, 'description': 'Bootstrap project for initializing the cloud.', 'name': self.project_name}
        PROVIDERS.resource_api.create_project(project_id, project)
        LOG.info('Created project %s', self.project_name)
    except exception.Conflict:
        LOG.info('Project %s already exists, skipping creation.', self.project_name)
        project = PROVIDERS.resource_api.get_project_by_name(self.project_name, self.default_domain_id)
    self.project_id = project['id']