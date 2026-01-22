from pprint import pformat
from six import iteritems
import re
@http_headers.setter
def http_headers(self, http_headers):
    """
        Sets the http_headers of this V1HTTPGetAction.
        Custom headers to set in the request. HTTP allows repeated headers.

        :param http_headers: The http_headers of this V1HTTPGetAction.
        :type: list[V1HTTPHeader]
        """
    self._http_headers = http_headers