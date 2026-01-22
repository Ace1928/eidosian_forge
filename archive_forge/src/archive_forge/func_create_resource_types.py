import random
from unittest import mock
import uuid
from openstack.image.v2 import _proxy
from openstack.image.v2 import cache
from openstack.image.v2 import image
from openstack.image.v2 import member
from openstack.image.v2 import metadef_namespace
from openstack.image.v2 import metadef_object
from openstack.image.v2 import metadef_property
from openstack.image.v2 import metadef_resource_type
from openstack.image.v2 import service_info as _service_info
from openstack.image.v2 import task
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def create_resource_types(attrs=None, count=2):
    """Create multiple fake resource types.

    :param attrs: A dictionary with all attributes of
        metadef_resource_type member
    :type attrs: dict
    :return: A list of fake MetadefResourceType objects
    :rtype: list
    """
    metadef_resource_types = []
    for n in range(0, count):
        metadef_resource_types.append(create_one_resource_type(attrs))
    return metadef_resource_types