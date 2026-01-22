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
def update_project_tags(self, project_id, tags, initiator=None):
    """Update all tags on a project.

        :param project_id: The ID of the project to update
        :param tags: A list of tags to update on the project

        :returns: A list of tags
        """
    project = self.driver.get_project(project_id)
    if ro_opt.check_resource_immutable(resource_ref=project):
        raise exception.ResourceUpdateForbidden(message=_('Cannot update project tags for %(project_id)s, project is immutable. Set "immutable" option to false before creating project tags.') % {'project_id': project_id})
    tag_list = [t.strip() for t in tags]
    project = {'tags': tag_list}
    self.update_project(project_id, project)
    return tag_list