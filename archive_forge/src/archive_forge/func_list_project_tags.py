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
def list_project_tags(self, project_id):
    """List all tags on project.

        :param project_id: The ID of a project

        :returns: A list of tags from a project
        """
    project = self.driver.get_project(project_id)
    return project.get('tags', [])