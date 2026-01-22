import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def get_current_versioned_data(self, session, allow=None, cache=None, project_id=None):
    """Run version discovery on the current endpoint.

        A simplified version of get_versioned_data, get_current_versioned_data
        runs discovery but only on the endpoint that has been found already.

        It can be useful in some workflows where the user wants version
        information about the endpoint they have.

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param dict allow: Extra filters to pass when discovering API
                           versions. (optional)
        :param dict cache: A dict to be used for caching results in
                           addition to caching them on the Session.
                           (optional)
        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)

        :returns: A new EndpointData with the requested versioned data.
        :rtype: :py:class:`keystoneauth1.discover.EndpointData`
        :raises keystoneauth1.exceptions.discovery.DiscoveryFailure: If the
                                                    appropriate versioned data
                                                    could not be discovered.
        """
    min_version, max_version = _normalize_version_args(self.api_version, None, None)
    return self.get_versioned_data(session=session, allow=allow, cache=cache, allow_version_hack=True, discover_versions=True, min_version=min_version, max_version=max_version)