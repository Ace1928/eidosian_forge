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
def search_resources(self, resource_type, name_or_id, get_args=None, get_kwargs=None, list_args=None, list_kwargs=None, **filters):
    """Search resources

        Search resources matching certain conditions

        :param str resource_type: String representation of the expected
            resource as `service.resource` (i.e. "network.security_group").
        :param str name_or_id: Name or ID of the resource
        :param list get_args: Optional args to be passed to the _get call.
        :param dict get_kwargs: Optional kwargs to be passed to the _get call.
        :param list list_args: Optional args to be passed to the _list call.
        :param dict list_kwargs: Optional kwargs to be passed to the _list call
        :param dict filters: Additional filters to be used for querying
            resources.
        """
    get_args = get_args or ()
    get_kwargs = get_kwargs or {}
    list_args = list_args or ()
    list_kwargs = list_kwargs or {}
    service_name, resource_name = resource_type.split('.')
    if not hasattr(self, service_name):
        raise exceptions.SDKException('service %s is not existing/enabled' % service_name)
    service_proxy = getattr(self, service_name)
    try:
        resource_type = service_proxy._resource_registry[resource_name]
    except KeyError:
        raise exceptions.SDKException('Resource %s is not known in service %s' % (resource_name, service_name))
    if name_or_id:
        try:
            resource_by_id = service_proxy._get(resource_type, name_or_id, *get_args, **get_kwargs)
            return [resource_by_id]
        except exceptions.ResourceNotFound:
            pass
    if not filters:
        filters = {}
    if name_or_id:
        filters['name'] = name_or_id
    list_kwargs.update(filters)
    return list(service_proxy._list(resource_type, *list_args, **list_kwargs))