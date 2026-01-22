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
def list_resource_provider_traits(self, resource_provider_uuid):
    """List all traits associated with a resource provider

        :param resource_provider_uuid: UUID of the resource provider for which
                                       the traits will be listed
        :raises PlacementResourceProviderNotFound: If the resource provider
                                                   is not found.
        :returns: The associated traits of the resource provider together
                  with the resource provider generation.
        """
    url = '/resource_providers/%s/traits' % resource_provider_uuid
    try:
        return self._get(url).json()
    except ks_exc.NotFound:
        raise n_exc.PlacementResourceProviderNotFound(resource_provider=resource_provider_uuid)