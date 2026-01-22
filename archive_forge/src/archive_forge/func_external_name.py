from pprint import pformat
from six import iteritems
import re
@external_name.setter
def external_name(self, external_name):
    """
        Sets the external_name of this V1ServiceSpec.
        externalName is the external reference that kubedns or equivalent will
        return as a CNAME record for this service. No proxying will be involved.
        Must be a valid RFC-1123 hostname (https://tools.ietf.org/html/rfc1123)
        and requires Type to be ExternalName.

        :param external_name: The external_name of this V1ServiceSpec.
        :type: str
        """
    self._external_name = external_name