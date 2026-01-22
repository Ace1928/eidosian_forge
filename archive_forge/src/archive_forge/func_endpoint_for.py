import copy
import functools
import queue
import warnings
import dogpile.cache
import keystoneauth1.exceptions
import keystoneauth1.session
import requests.models
import requestsexceptions
from openstack import _log
from openstack.cloud import _object_store
from openstack.cloud import _utils
from openstack.cloud import meta
import openstack.config
from openstack.config import cloud_region as cloud_region_mod
from openstack import exceptions
from openstack import proxy
from openstack import utils
def endpoint_for(self, service_type, interface=None, region_name=None):
    """Return the endpoint for a given service.

        Respects config values for Connection, including
        ``*_endpoint_override``. For direct values from the catalog
        regardless of overrides, see
        :meth:`~openstack.config.cloud_region.CloudRegion.get_endpoint_from_catalog`

        :param service_type: Service Type of the endpoint to search for.
        :param interface:
            Interface of the endpoint to search for. Optional, defaults to
            the configured value for interface for this Connection.
        :param region_name:
            Region Name of the endpoint to search for. Optional, defaults to
            the configured value for region_name for this Connection.

        :returns: The endpoint of the service, or None if not found.
        """
    endpoint_override = self.config.get_endpoint(service_type)
    if endpoint_override:
        return endpoint_override
    return self.config.get_endpoint_from_catalog(service_type=service_type, interface=interface, region_name=region_name)