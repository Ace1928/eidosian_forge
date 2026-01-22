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
def create_one_metadef_object(attrs=None):
    """Create a fake MetadefNamespace member.

    :param attrs: A dictionary with all attributes of metadef_namespace member
    :type attrs: dict
    :return: a list of MetadefNamespace objects
    :rtype: list of `metadef_namespace.MetadefNamespace`
    """
    attrs = attrs or {}
    metadef_objects_list = {'created_at': '2014-09-19T18:20:56Z', 'description': 'The CPU limits with control parameters.', 'name': 'CPU Limits', 'properties': {'quota:cpu_period': {'description': 'The enforcement interval', 'maximum': 1000000, 'minimum': 1000, 'title': 'Quota: CPU Period', 'type': 'integer'}, 'quota:cpu_quota': {'description': 'The maximum allowed bandwidth', 'title': 'Quota: CPU Quota', 'type': 'integer'}, 'quota:cpu_shares': {'description': 'The proportional weighted', 'title': 'Quota: CPU Shares', 'type': 'integer'}}, 'required': [], 'schema': '/v2/schemas/metadefs/object', 'updated_at': '2014-09-19T18:20:56Z'}
    metadef_objects_list.update(attrs)
    return metadef_object.MetadefObject(**metadef_objects_list)