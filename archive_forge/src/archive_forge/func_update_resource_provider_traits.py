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
def update_resource_provider_traits(self, resource_provider_uuid, traits, resource_provider_generation=None):
    """Replace all associated traits of a resource provider.

        :param resource_provider_uuid: UUID of the resource provider for which
                                       to set the traits
        :param traits: a list of traits.
        :param resource_provider_generation: The generation of the resource
                                             provider. Optional. If not
                                             supplied by the caller, handle
                                             potential generation conflict
                                             by retrying the call. If supplied
                                             we assume the caller handles
                                             generation conflict.
        :raises PlacementResourceProviderNotFound: If the resource provider
                                                   is not found.
        :raises PlacementTraitNotFound: If any of the specified traits are not
                                        valid.
        :raises PlacementResourceProviderGenerationConflict: For concurrent
                                                             conflicting
                                                             updates detected.
        :returns: The new traits of the resource provider together with the
                  resource provider generation.
        """
    url = '/resource_providers/%s/traits' % resource_provider_uuid
    body = {'resource_provider_generation': resource_provider_generation, 'traits': traits}
    try:
        return self._put_with_retry_for_generation_conflict(url, body, resource_provider_uuid, resource_provider_generation)
    except ks_exc.NotFound:
        raise n_exc.PlacementResourceProviderNotFound(resource_provider=resource_provider_uuid)
    except ks_exc.BadRequest:
        raise n_exc.PlacementTraitNotFound(trait=traits)