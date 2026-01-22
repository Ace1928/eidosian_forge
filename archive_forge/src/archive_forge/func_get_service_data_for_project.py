import copy
import os_service_types.data
from os_service_types import exc
def get_service_data_for_project(self, project_name):
    """Return the service information associated with a project.

        :param name: A repository or project name in the form
            ``'openstack/{project}'`` or just ``'{project}'``.
        :type name: str
        :raises ValueError: If project_name is None
        :returns: dict or None if not found
        """
    project_name = self._canonical_project_name(project_name)
    return self.primary_service_by_project.get(project_name)