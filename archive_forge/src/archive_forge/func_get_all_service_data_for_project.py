import copy
import os_service_types.data
from os_service_types import exc
def get_all_service_data_for_project(self, project_name):
    """Return the information for every service associated with a project.

        :param name: A repository or project name in the form
            ``'openstack/{project}'`` or just ``'{project}'``.
        :type name: str
        :raises ValueError: If project_name is None
        :returns: list of dicts
        """
    data = []
    for service_type in self.service_types_by_project.get(project_name, []):
        data.append(self.get_service_data(service_type))
    return data