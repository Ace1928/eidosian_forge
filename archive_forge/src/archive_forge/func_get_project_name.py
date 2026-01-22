import copy
import os_service_types.data
from os_service_types import exc
def get_project_name(self, service_type):
    """Return the OpenStack project name for a given service_type.

        :param str service_type: An official service-type or alias.
        :returns str: The OpenStack project name or None if there is no match.
        """
    service_type = _normalize_type(service_type)
    service = self.get_service_data(service_type)
    if service:
        return service['project']
    return None