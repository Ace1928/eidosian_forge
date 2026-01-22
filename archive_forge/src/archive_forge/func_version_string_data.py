import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def version_string_data(self, reverse=False, **kwargs):
    """Get normalized version data with versions as strings.

        Return version data in a structured way.

        :param bool reverse: Reverse the list. reverse=true will mean the
                             returned list is sorted from newest to oldest
                             version.
        :returns: A list of :class:`VersionData` sorted by version number.
        :rtype: list(VersionData)
        """
    version_data = self.version_data(reverse=reverse, **kwargs)
    for version in version_data:
        for key in ('version', 'min_microversion', 'max_microversion'):
            if version[key]:
                version[key] = version_to_string(version[key])
    return version_data