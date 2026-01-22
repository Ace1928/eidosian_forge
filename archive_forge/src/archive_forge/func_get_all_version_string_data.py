import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def get_all_version_string_data(self, session, project_id=None):
    """Return version data for all versions discovery can find.

        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)
        :returns: A list of :class:`VersionData` sorted by version number.
        :rtype: list(VersionData)
        """
    versions = []
    for vers_url in self._get_discovery_url_choices(project_id=project_id):
        try:
            d = get_discovery(session, vers_url)
        except Exception as e:
            _LOGGER.debug('Failed attempt at discovery on %s: %s', vers_url, str(e))
            continue
        for version in d.version_string_data():
            versions.append(version)
        break
    return versions or self._infer_version_data(project_id)