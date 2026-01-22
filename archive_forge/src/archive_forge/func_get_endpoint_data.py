import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def get_endpoint_data(self, session, service_type=None, interface=None, region_name=None, service_name=None, allow=None, allow_version_hack=True, discover_versions=True, skip_discovery=False, min_version=None, max_version=None, endpoint_override=None, **kwargs):
    """Return a valid endpoint data for a service.

        If a valid token is not present then a new one will be fetched using
        the session and kwargs.

        version, min_version and max_version can all be given either as a
        string or a tuple.

        Valid interface types: `public` or `publicURL`,
                               `internal` or `internalURL`,
                               `admin` or 'adminURL`

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param string service_type: The type of service to lookup the endpoint
                                    for. This plugin will return None (failure)
                                    if service_type is not provided.
        :param interface: Type of endpoint. Can be a single value or a list
                          of values. If it's a list of values, they will be
                          looked for in order of preference. Can also be
                          `keystoneauth1.plugin.AUTH_INTERFACE` to indicate
                          that the auth_url should be used instead of the
                          value in the catalog. (optional, defaults to public)
        :param string region_name: The region the endpoint should exist in.
                                   (optional)
        :param string service_name: The name of the service in the catalog.
                                   (optional)
        :param dict allow: Extra filters to pass when discovering API
                           versions. (optional)
        :param bool allow_version_hack: Allow keystoneauth to hack up catalog
                                        URLS to support older schemes.
                                        (optional, default True)
        :param bool discover_versions: Whether to get version metadata from
                                       the version discovery document even
                                       if it's not neccessary to fulfill the
                                       major version request. (optional,
                                       defaults to True)
        :param bool skip_discovery: Whether to skip version discovery even
                                    if a version has been given. This is useful
                                    if endpoint_override or similar has been
                                    given and grabbing additional information
                                    about the endpoint is not useful.
        :param min_version: The minimum version that is acceptable. Mutually
                            exclusive with version. If min_version is given
                            with no max_version it is as if max version is
                            'latest'. (optional)
        :param max_version: The maximum version that is acceptable. Mutually
                            exclusive with version. If min_version is given
                            with no max_version it is as if max version is
                            'latest'. (optional)
        :param str endpoint_override: URL to use instead of looking in the
                                      catalog. Catalog lookup will be skipped,
                                      but version discovery will be run.
                                      Sets allow_version_hack to False
                                      (optional)
        :param kwargs: Ignored.

        :raises keystoneauth1.exceptions.http.HttpError: An error from an
                                                         invalid HTTP response.

        :return: Valid EndpointData or None if not available.
        :rtype: `keystoneauth1.discover.EndpointData` or None
        """
    allow = allow or {}
    min_version, max_version = discover._normalize_version_args(None, min_version, max_version, service_type=service_type)
    if interface is plugin.AUTH_INTERFACE:
        endpoint_data = discover.EndpointData(service_url=self.auth_url, service_type=service_type or 'identity')
        project_id = None
    elif endpoint_override:
        endpoint_data = discover.EndpointData(service_url=endpoint_override, catalog_url=endpoint_override, interface=interface, region_name=region_name, service_name=service_name)
        allow_version_hack = False
        project_id = self.get_project_id(session)
    else:
        if not service_type:
            LOG.warning('Plugin cannot return an endpoint without knowing the service type that is required. Add service_type to endpoint filtering data.')
            return None
        if not interface:
            interface = 'public'
        service_catalog = self.get_access(session).service_catalog
        project_id = self.get_project_id(session)
        endpoint_data = service_catalog.endpoint_data_for(service_type=service_type, interface=interface, region_name=region_name, service_name=service_name)
        if not endpoint_data:
            return None
    if skip_discovery:
        return endpoint_data
    try:
        return endpoint_data.get_versioned_data(session, project_id=project_id, min_version=min_version, max_version=max_version, cache=self._discovery_cache, discover_versions=discover_versions, allow_version_hack=allow_version_hack, allow=allow)
    except (exceptions.DiscoveryFailure, exceptions.HttpError, exceptions.ConnectionError):
        if max_version or min_version:
            return None
        return endpoint_data