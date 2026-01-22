import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
@_check_placement_api_available
def list_resource_providers(self, name=None, member_of=None, resources=None, in_tree=None, uuid=None):
    """Get a list of resource providers.

        :param name: Name of the resource providers.
        :param member_of: List of aggregate UUID to get those resource
                          providers that are associated with.
                          NOTE: placement 1.3 needed.
        :param resources: Dictionary of resource classes and requested values.
        :param in_tree: UUID of a resource provider that the caller wants to
                        limit the returned providers to those within its
                        'provider tree'. The returned list will contain only
                        resource providers with the root_provider_id of the
                        resource provider with UUID == tree_uuid.
                        NOTE: placement 1.14 needed.
        :param uuid: UUID of the resource provider.
        :raises PlacementAPIVersionIncorrect: If placement API target version
                                              is too low
        :returns: A list of Resource Provider matching the filters.
        """
    url = '/resource_providers'
    filters = {}
    if name:
        filters['name'] = name
    if member_of:
        needed_version = _get_version(PLACEMENT_API_WITH_MEMBER_OF)
        if self._target_version < needed_version:
            raise n_exc.PlacementAPIVersionIncorrect(current_version=self._target_version, needed_version=needed_version)
        filters['member_of'] = member_of
    if resources:
        filters['resources'] = resources
    if in_tree:
        needed_version = _get_version(PLACEMENT_API_WITH_NESTED_RESOURCES)
        if self._target_version < needed_version:
            raise n_exc.PlacementAPIVersionIncorrect(current_version=self._target_version, needed_version=needed_version)
        filters['in_tree'] = in_tree
    if uuid:
        filters['uuid'] = uuid
    url = '%s?%s' % (url, parse.urlencode(filters))
    return self._get(url).json()