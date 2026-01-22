import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def versioned_data_for(self, url=None, min_version=None, max_version=None, **kwargs):
    """Return endpoint data for the service at a url.

        min_version and max_version can be given either as strings or tuples.

        :param string url: If url is given, the data will be returned for the
            endpoint data that has a self link matching the url.
        :param min_version: The minimum endpoint version that is acceptable. If
            min_version is given with no max_version it is as if max version is
            'latest'. If min_version is 'latest', max_version may only be
            'latest' or None.
        :param max_version: The maximum endpoint version that is acceptable. If
            min_version is given with no max_version it is as if max version is
            'latest'. If min_version is 'latest', max_version may only be
            'latest' or None.

        :returns: the endpoint data for a URL that matches the required version
                  (the format is described in version_data) or None if no
                  match.
        :rtype: dict
        """
    min_version, max_version = _normalize_version_args(None, min_version, max_version)
    no_version = not max_version and (not min_version)
    version_data = self.version_data(reverse=True, **kwargs)
    if max_version == (LATEST, LATEST) and (not min_version or min_version == (LATEST, LATEST)):
        return version_data[0]
    if url:
        url = url.rstrip('/') + '/'
    if no_version and (not url):
        return version_data[0]
    for data in version_data:
        if url and data['url'] and (data['url'].rstrip('/') + '/' == url):
            return data
        if _latest_soft_match(min_version, data['version']):
            return data
        if min_version and max_version and version_between(min_version, max_version, data['version']):
            return data
    if no_version and url and (len(version_data) > 0):
        return version_data[0]
    return None