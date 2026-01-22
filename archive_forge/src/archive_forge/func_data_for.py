import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def data_for(self, version, **kwargs):
    """Return endpoint data for a version.

        NOTE: This method raises a TypeError if version is None. It is
              kept for backwards compatability. New code should use
              versioned_data_for instead.

        :param tuple version: The version is always a minimum version in the
            same major release as there should be no compatibility issues with
            using a version newer than the one asked for.

        :returns: the endpoint data for a URL that matches the required version
                  (the format is described in version_data) or None if no
                  match.
        :rtype: dict
        """
    version = normalize_version_number(version)
    for data in self.version_data(reverse=True, **kwargs):
        if _latest_soft_match(version, data['version']):
            return data
        if version_match(version, data['version']):
            return data
    return None