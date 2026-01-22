import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
@instance_cache
def load_service_model(self, service_name, type_name, api_version=None):
    """Load a botocore service model

        This is the main method for loading botocore models (e.g. a service
        model, pagination configs, waiter configs, etc.).

        :type service_name: str
        :param service_name: The name of the service (e.g ``ec2``, ``s3``).

        :type type_name: str
        :param type_name: The model type.  Valid types include, but are not
            limited to: ``service-2``, ``paginators-1``, ``waiters-2``.

        :type api_version: str
        :param api_version: The API version to load.  If this is not
            provided, then the latest API version will be used.

        :type load_extras: bool
        :param load_extras: Whether or not to load the tool extras which
            contain additional data to be added to the model.

        :raises: UnknownServiceError if there is no known service with
            the provided service_name.

        :raises: DataNotFoundError if no data could be found for the
            service_name/type_name/api_version.

        :return: The loaded data, as a python type (e.g. dict, list, etc).
        """
    known_services = self.list_available_services(type_name)
    if service_name not in known_services:
        raise UnknownServiceError(service_name=service_name, known_service_names=', '.join(sorted(known_services)))
    if api_version is None:
        api_version = self.determine_latest_version(service_name, type_name)
    full_path = os.path.join(service_name, api_version, type_name)
    model = self.load_data(full_path)
    extras_data = self._find_extras(service_name, type_name, api_version)
    self._extras_processor.process(model, extras_data)
    return model