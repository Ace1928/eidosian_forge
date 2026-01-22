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
def create_one_stores_info(attrs=None):
    """Create a fake stores info.

    :param attrs: A dictionary with all attributes of stores
    :type attrs: dict
    :return: A fake Store object list.
    :rtype: `openstack.image.v2.service_info.Store`
    """
    attrs = attrs or {}
    stores_info = {'stores': [{'id': 'reliable', 'description': 'More expensive store with data redundancy'}, {'id': 'fast', 'description': 'Provides quick access to your image data', 'default': True}, {'id': 'cheap', 'description': 'Less expensive store for seldom-used images'}]}
    stores_info.update(attrs)
    return _service_info.Store(**stores_info)