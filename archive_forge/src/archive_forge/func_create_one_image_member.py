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
def create_one_image_member(attrs=None):
    """Create a fake image member.

    :param attrs: A dictionary with all attributes of image member
    :type attrs: dict
    :return: A fake Member object.
    :rtype: `openstack.image.v2.member.Member`
    """
    attrs = attrs or {}
    image_member_info = {'member_id': 'member-id-' + uuid.uuid4().hex, 'image_id': 'image-id-' + uuid.uuid4().hex, 'status': 'pending'}
    image_member_info.update(attrs)
    return member.Member(**image_member_info)