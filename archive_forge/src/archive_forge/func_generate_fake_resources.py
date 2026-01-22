import inspect
import random
from typing import (
from unittest import mock
import uuid
from openstack import format as _format
from openstack import proxy
from openstack import resource
from openstack import service_description
def generate_fake_resources(resource_type: Type[Resource], count: int=1, attrs: Optional[Dict[str, Any]]=None) -> Generator[Resource, None, None]:
    """Generate a given number of fake resource entities

    :param type resource_type: Object class
    :param int count: Number of objects to return
    :param dict attrs: Attribute values to set into each instance

    Example usage:

    .. code-block:: python

        >>> from openstack.compute.v2 import server
        >>> from openstack.test import fakes
        >>> fakes.generate_fake_resources(server.Server, count=3)
        <generator object generate_fake_resources at 0x7f075dc65040>

    :param type resource_type: Object class
    :param int count: Number of objects to return
    :param dict attrs: Attribute values to set into each instance
    :return: Generator of ``resource_type`` class instances populated with fake
        values of expected types.
    """
    if not attrs:
        attrs = {}
    for _ in range(count):
        yield generate_fake_resource(resource_type, **attrs)