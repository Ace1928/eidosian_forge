import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_one_service(attrs=None):
    """Create a fake service.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, type, and so on
        """
    attrs = attrs or {}
    service_info = {'id': 'service-id-' + uuid.uuid4().hex, 'name': 'service-name-' + uuid.uuid4().hex, 'description': 'service_description', 'type': 'service_type'}
    service_info.update(attrs)
    service = fakes.FakeResource(info=copy.deepcopy(service_info), loaded=True)
    return service