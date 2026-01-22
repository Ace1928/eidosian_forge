from pprint import pformat
from six import iteritems
import re
@preferred_version.setter
def preferred_version(self, preferred_version):
    """
        Sets the preferred_version of this V1APIGroup.
        preferredVersion is the version preferred by the API server, which
        probably is the storage version.

        :param preferred_version: The preferred_version of this V1APIGroup.
        :type: V1GroupVersionForDiscovery
        """
    self._preferred_version = preferred_version