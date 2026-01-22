from pprint import pformat
from six import iteritems
import re
@enable_service_links.setter
def enable_service_links(self, enable_service_links):
    """
        Sets the enable_service_links of this V1PodSpec.
        EnableServiceLinks indicates whether information about services should
        be injected into pod's environment variables, matching the syntax of
        Docker links. Optional: Defaults to true.

        :param enable_service_links: The enable_service_links of this V1PodSpec.
        :type: bool
        """
    self._enable_service_links = enable_service_links