from pprint import pformat
from six import iteritems
import re
@tenant.setter
def tenant(self, tenant):
    """
        Sets the tenant of this V1QuobyteVolumeSource.
        Tenant owning the given Quobyte volume in the Backend Used with
        dynamically provisioned Quobyte volumes, value is set by the plugin

        :param tenant: The tenant of this V1QuobyteVolumeSource.
        :type: str
        """
    self._tenant = tenant