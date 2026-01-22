from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoint_groups
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import projects
def delete_endpoint_group_from_project(self, endpoint_group, project):
    """Remove a project-endpoint group association."""
    if not (project and endpoint_group):
        raise ValueError(_('project and endpoint_group are required'))
    base_url = self._build_group_base_url(project=project, endpoint_group=endpoint_group)
    return super(EndpointFilterManager, self)._delete(url=base_url)