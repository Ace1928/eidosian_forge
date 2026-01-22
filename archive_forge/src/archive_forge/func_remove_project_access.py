from cinderclient.apiclient import base as common_base
from cinderclient import base
def remove_project_access(self, volume_type, project):
    """Remove a project from the given volume type access list."""
    info = {'project': project}
    return self._action('removeProjectAccess', volume_type, info)