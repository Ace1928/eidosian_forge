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
def create_one_image(attrs=None):
    """Create a fake image.

    :param attrs: A dictionary with all attributes of image
    :type attrs: dict
    :return: A fake Image object.
    :rtype: `openstack.image.v2.image.Image`
    """
    attrs = attrs or {}
    image_info = {'id': str(uuid.uuid4()), 'name': 'image-name' + uuid.uuid4().hex, 'owner_id': 'image-owner' + uuid.uuid4().hex, 'is_protected': bool(random.choice([0, 1])), 'visibility': random.choice(['public', 'private']), 'tags': [uuid.uuid4().hex for r in range(2)]}
    image_info.update(attrs)
    return image.Image(**image_info)