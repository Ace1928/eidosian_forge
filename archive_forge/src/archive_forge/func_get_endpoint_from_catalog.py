import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def get_endpoint_from_catalog(self, service_type, interface=None, region_name=None):
    """Return the endpoint for a given service as found in the catalog.

        For values respecting endpoint overrides, see
        :meth:`~openstack.connection.Connection.endpoint_for`

        :param service_type: Service Type of the endpoint to search for.
        :param interface:
            Interface of the endpoint to search for. Optional, defaults to
            the configured value for interface for this Connection.
        :param region_name:
            Region Name of the endpoint to search for. Optional, defaults to
            the configured value for region_name for this Connection.

        :returns: The endpoint of the service, or None if not found.
        """
    interface = interface or self.get_interface(service_type)
    region_name = region_name or self.get_region_name(service_type)
    session = self.get_session()
    catalog = session.auth.get_access(session).service_catalog
    try:
        return catalog.url_for(service_type=service_type, interface=interface, region_name=region_name)
    except (keystoneauth1.exceptions.catalog.EndpointNotFound, ValueError):
        return None