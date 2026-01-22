import copy
import os_service_types.data
from os_service_types import exc
@property
def service_types_by_project(self):
    """Mapping of project name to a list of all associated service-types."""
    return copy.deepcopy(self._service_types_data['service_types_by_project'])