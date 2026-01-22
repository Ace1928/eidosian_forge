import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
@instance_cache
def list_available_services(self, type_name):
    """List all known services.

        This will traverse the search path and look for all known
        services.

        :type type_name: str
        :param type_name: The type of the service (service-2,
            paginators-1, waiters-2, etc).  This is needed because
            the list of available services depends on the service
            type.  For example, the latest API version available for
            a resource-1.json file may not be the latest API version
            available for a services-2.json file.

        :return: A list of all services.  The list of services will
            be sorted.

        """
    services = set()
    for possible_path in self._potential_locations():
        possible_services = [d for d in os.listdir(possible_path) if os.path.isdir(os.path.join(possible_path, d))]
        for service_name in possible_services:
            full_dirname = os.path.join(possible_path, service_name)
            api_versions = os.listdir(full_dirname)
            for api_version in api_versions:
                full_load_path = os.path.join(full_dirname, api_version, type_name)
                if self.file_loader.exists(full_load_path):
                    services.add(service_name)
                    break
    return sorted(services)