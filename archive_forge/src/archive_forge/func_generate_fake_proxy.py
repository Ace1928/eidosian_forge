import inspect
import random
from typing import (
from unittest import mock
import uuid
from openstack import format as _format
from openstack import proxy
from openstack import resource
from openstack import service_description
def generate_fake_proxy(service: Type[service_description.ServiceDescription], api_version: Optional[str]=None) -> proxy.Proxy:
    """Generate a fake proxy for the given service type

    Example usage:

    .. code-block:: python

        >>> import functools
        >>> from openstack.compute import compute_service
        >>> from openstack.compute.v2 import server
        >>> from openstack.test import fakes
        >>> # create the fake proxy
        >>> fake_compute_proxy = fakes.generate_fake_proxy(
        ...    compute_service.ComputeService,
        ... )
        >>> # configure return values for various proxy APIs
        >>> # note that this will generate new fake resources on each invocation
        >>> fake_compute_proxy.get_server.side_effect = functools.partial(
        ...     fakes.generate_fake_resource,
        ...     server.Server,
        ... )
        >>> fake_compute_proxy.servers.side_effect = functools.partial(
        ...     fakes.generate_fake_resources,
        ...     server.Server,
        ... )
        >>> fake_compute_proxy.servers()
        <generator object generate_fake_resources at 0x7f92768dc040>
        >>> fake_compute_proxy.serverssss()
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/usr/lib64/python3.11/unittest/mock.py", line 653, in __getattr__
            raise AttributeError("Mock object has no attribute %r" % name)
        AttributeError: Mock object has no attribute 'serverssss'. Did you mean: 'server_ips'?

    :param service: The service to generate the fake proxy for.
    :type service: :class:`~openstack.service_description.ServiceDescription`
    :param api_version: The API version to generate the fake proxy for.
        This should be a major version must be supported by openstacksdk, as
        specified in the ``supported_versions`` attribute of the provided
        ``service``. This is only required if openstacksdk supports multiple
        API versions for the given service.
    :type api_version: int or None
    :raises ValueError: if the ``service`` is not a valid
        :class:`~openstack.service_description.ServiceDescription` or if
        ``api_version`` is not supported
    :returns: An autospecced mock of the :class:`~openstack.proxy.Proxy`
        implementation for the specified service type and API version
    """
    if not issubclass(service, service_description.ServiceDescription):
        raise ValueError(f'Service {service.__name__} is not a valid ServiceDescription')
    supported_versions = service.supported_versions
    if api_version is None:
        if len(supported_versions) > 1:
            raise ValueError(f'api_version was not provided but service {service.__name__} provides multiple API versions')
        else:
            api_version = list(supported_versions)[0]
    elif api_version not in supported_versions:
        raise ValueError(f'API version {api_version} is not supported by openstacksdk. Supported API versions are: {', '.join(supported_versions)}')
    return mock.create_autospec(supported_versions[api_version])