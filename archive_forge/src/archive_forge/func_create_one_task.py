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
def create_one_task(attrs=None):
    """Create a fake task.

    :param attrs: A dictionary with all attributes of task
    :type attrs: dict
    :return: A fake Task object.
    :rtype: `openstack.image.v2.task.Task`
    """
    attrs = attrs or {}
    task_info = {'created_at': '2016-06-29T16:13:07Z', 'expires_at': '2016-07-01T16:13:07Z', 'id': str(uuid.uuid4()), 'input': {'image_properties': {'container_format': 'ovf', 'disk_format': 'vhd'}, 'import_from': 'https://apps.openstack.org/excellent-image', 'import_from_format': 'qcow2'}, 'message': '', 'owner': str(uuid.uuid4()), 'result': {'image_id': str(uuid.uuid4())}, 'schema': '/v2/schemas/task', 'status': random.choice(['pending', 'processing', 'success', 'failure']), 'type': 'import', 'updated_at': '2016-06-29T16:13:07Z'}
    task_info.update(attrs)
    return task.Task(**task_info)