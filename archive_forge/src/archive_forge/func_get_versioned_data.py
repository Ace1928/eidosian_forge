import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def get_versioned_data(self, session, allow=None, cache=None, allow_version_hack=True, project_id=None, discover_versions=True, min_version=None, max_version=None):
    """Run version discovery for the service described.

        Performs Version Discovery and returns a new EndpointData object with
        information found.

        min_version and max_version can be given either as strings or tuples.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param dict allow: Extra filters to pass when discovering API
                           versions. (optional)
        :param dict cache: A dict to be used for caching results in
                           addition to caching them on the Session.
                           (optional)
        :param bool allow_version_hack: Allow keystoneauth to hack up catalog
                                        URLS to support older schemes.
                                        (optional, default True)
        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)
        :param bool discover_versions: Whether to get version metadata from
                                       the version discovery document even
                                       if it's not neccessary to fulfill the
                                       major version request. (optional,
                                       defaults to True)
        :param min_version: The minimum version that is acceptable. If
                            min_version is given with no max_version it is as
                            if max version is 'latest'.
        :param max_version: The maximum version that is acceptable. If
                            min_version is given with no max_version it is as
                            if max version is 'latest'.

        :returns: A new EndpointData with the requested versioned data.
        :rtype: :py:class:`keystoneauth1.discover.EndpointData`
        :raises keystoneauth1.exceptions.discovery.DiscoveryFailure: If the
                                                    appropriate versioned data
                                                    could not be discovered.
        """
    min_version, max_version = _normalize_version_args(None, min_version, max_version)
    if not allow:
        allow = {}
    new_data = copy.copy(self)
    new_data._set_version_info(session=session, allow=allow, cache=cache, allow_version_hack=allow_version_hack, project_id=project_id, discover_versions=discover_versions, min_version=min_version, max_version=max_version)
    return new_data