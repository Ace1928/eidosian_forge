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
def get_session_client(self, service_type, version=None, constructor=proxy.Proxy, **kwargs):
    """Return a prepped keystoneauth Adapter for a given service.

        This is useful for making direct requests calls against a
        'mounted' endpoint. That is, if you do:

          client = get_session_client('compute')

        then you can do:

          client.get('/flavors')

        and it will work like you think.
        """
    version_request = self._get_version_request(service_type, version)
    kwargs.setdefault('region_name', self.get_region_name(service_type))
    kwargs.setdefault('connect_retries', self.get_connect_retries(service_type))
    kwargs.setdefault('status_code_retries', self.get_status_code_retries(service_type))
    kwargs.setdefault('statsd_prefix', self.get_statsd_prefix())
    kwargs.setdefault('statsd_client', self.get_statsd_client())
    kwargs.setdefault('prometheus_counter', self.get_prometheus_counter())
    kwargs.setdefault('prometheus_histogram', self.get_prometheus_histogram())
    kwargs.setdefault('influxdb_config', self._influxdb_config)
    kwargs.setdefault('influxdb_client', self.get_influxdb_client())
    endpoint_override = self.get_endpoint(service_type)
    version = version_request.version
    min_api_version = kwargs.pop('min_version', None) or version_request.min_api_version
    max_api_version = kwargs.pop('max_version', None) or version_request.max_api_version
    if service_type in ('network', 'load-balancer'):
        version = None
        min_api_version = None
        max_api_version = None
        if endpoint_override is None:
            endpoint_override = self._get_hardcoded_endpoint(service_type, constructor)
    client = constructor(session=self.get_session(), service_type=self.get_service_type(service_type), service_name=self.get_service_name(service_type), interface=self.get_interface(service_type), version=version, min_version=min_api_version, max_version=max_api_version, endpoint_override=endpoint_override, default_microversion=version_request.default_microversion, rate_limit=self.get_rate_limit(service_type), concurrency=self.get_concurrency(service_type), **kwargs)
    if version_request.default_microversion:
        default_microversion = version_request.default_microversion
        info = client.get_endpoint_data()
        if not discover.version_between(info.min_microversion, info.max_microversion, default_microversion):
            if self.get_default_microversion(service_type):
                raise exceptions.ConfigException('A default microversion for service {service_type} of {default_microversion} was requested, but the cloud only supports a minimum of {min_microversion} and a maximum of {max_microversion}.'.format(service_type=service_type, default_microversion=default_microversion, min_microversion=discover.version_to_string(info.min_microversion), max_microversion=discover.version_to_string(info.max_microversion)))
            else:
                raise exceptions.ConfigException("A default microversion for service {service_type} of {default_microversion} was requested, but the cloud only supports a minimum of {min_microversion} and a maximum of {max_microversion}. The default microversion was set because a microversion formatted version string, '{api_version}', was passed for the api_version of the service. If it was not intended to set a default microversion please remove anything other than an integer major version from the version setting for the service.".format(service_type=service_type, api_version=self.get_api_version(service_type), default_microversion=default_microversion, min_microversion=discover.version_to_string(info.min_microversion), max_microversion=discover.version_to_string(info.max_microversion)))
    return client