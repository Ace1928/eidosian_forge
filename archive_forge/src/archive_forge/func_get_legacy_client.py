import importlib
from openstack import _log
from openstack.config import cloud_region
import os_client_config
from os_client_config import constructors
from os_client_config import exceptions
def get_legacy_client(self, service_key, client_class=None, interface_key=None, pass_version_arg=True, version=None, min_version=None, max_version=None, **kwargs):
    """Return a legacy OpenStack client object for the given config.

        Most of the OpenStack python-*client libraries have the same
        interface for their client constructors, but there are several
        parameters one wants to pass given a :class:`CloudConfig` object.

        In the future, OpenStack API consumption should be done through
        the OpenStack SDK, but that's not ready yet. This is for getting
        Client objects from python-*client only.

        :param service_key: Generic key for service, such as 'compute' or
                            'network'
        :param client_class: Class of the client to be instantiated. This
                             should be the unversioned version if there
                             is one, such as novaclient.client.Client, or
                             the versioned one, such as
                             neutronclient.v2_0.client.Client if there isn't
        :param interface_key: (optional) Some clients, such as glanceclient
                              only accept the parameter 'interface' instead
                              of 'endpoint_type' - this is a get-out-of-jail
                              parameter for those until they can be aligned.
                              os-client-config understands this to be the
                              case if service_key is image, so this is really
                              only for use with other unknown broken clients.
        :param pass_version_arg: (optional) If a versioned Client constructor
                                 was passed to client_class, set this to
                                 False, which will tell get_client to not
                                 pass a version parameter. os-client-config
                                 already understand that this is the
                                 case for network, so it can be omitted in
                                 that case.
        :param version: (optional) Version string to override the configured
                                   version string.
        :param min_version: (options) Minimum version acceptable.
        :param max_version: (options) Maximum version acceptable.
        :param kwargs: (optional) keyword args are passed through to the
                       Client constructor, so this is in case anything
                       additional needs to be passed in.
        """
    if not client_class:
        client_class = _get_client(service_key)
    interface = self.get_interface(service_key)
    endpoint = self.get_session_endpoint(service_key, min_version=min_version, max_version=max_version)
    endpoint_override = self.get_endpoint(service_key)
    if service_key == 'object-store':
        constructor_kwargs = dict(session=self.get_session(), os_options=dict(service_type=self.get_service_type(service_key), object_storage_url=endpoint_override, region_name=self.region))
    else:
        constructor_kwargs = dict(session=self.get_session(), service_name=self.get_service_name(service_key), service_type=self.get_service_type(service_key), endpoint_override=endpoint_override, region_name=self.region)
    if service_key == 'image':
        from glanceclient.common import utils as glance_utils
        endpoint, detected_version = glance_utils.strip_version(endpoint)
        if not version:
            version = detected_version
        if not endpoint_override:
            constructor_kwargs['endpoint_override'] = endpoint
    constructor_kwargs.update(kwargs)
    if pass_version_arg and service_key != 'object-store':
        if not version:
            version = self.get_api_version(service_key)
        if not version and service_key == 'volume':
            from cinderclient import client as cinder_client
            version = cinder_client.get_volume_api_from_url(endpoint)
        if service_key == 'network' and version == '2':
            version = '2.0'
        if service_key == 'identity':
            if 'endpoint' not in constructor_kwargs:
                endpoint = self.get_session_endpoint('identity')
                constructor_kwargs['endpoint'] = endpoint
        if service_key == 'network':
            constructor_kwargs['api_version'] = version
        elif service_key == 'baremetal':
            if version != '1':
                constructor_kwargs['os_ironic_api_version'] = version
            constructor_kwargs['version'] = version[0]
        else:
            constructor_kwargs['version'] = version
        if min_version and min_version > float(version):
            raise exceptions.OpenStackConfigVersionException('Minimum version {min_version} requested but {version} found'.format(min_version=min_version, version=version), version=version)
        if max_version and max_version < float(version):
            raise exceptions.OpenStackConfigVersionException('Maximum version {max_version} requested but {version} found'.format(max_version=max_version, version=version), version=version)
    if service_key == 'database':
        constructor_kwargs['username'] = None
        constructor_kwargs['password'] = None
    if not interface_key:
        if service_key in ('image', 'key-manager'):
            interface_key = 'interface'
        elif service_key == 'identity' and version and version.startswith('3'):
            interface_key = 'interface'
        else:
            interface_key = 'endpoint_type'
    if service_key == 'object-store':
        constructor_kwargs['os_options'][interface_key] = interface
    else:
        constructor_kwargs[interface_key] = interface
    return client_class(**constructor_kwargs)