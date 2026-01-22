import copy
import os_service_types.data
from os_service_types import exc
def is_official(self, service_type):
    """Is the given service-type an official service-type?

        :param str service_type: The service-type to test.
        :returns bool: True if it's an official type, False otherwise.
        """
    return self.get_official_service_data(service_type) is not None