import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_one_endpoint(attrs=None):
    """Create a fake agent.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, region, and so on
        """
    attrs = attrs or {}
    endpoint_info = {'service_name': 'service-name-' + uuid.uuid4().hex, 'adminurl': 'http://endpoint_adminurl', 'region': 'endpoint_region', 'internalurl': 'http://endpoint_internalurl', 'service_type': 'service_type', 'id': 'endpoint-id-' + uuid.uuid4().hex, 'publicurl': 'http://endpoint_publicurl', 'service_id': 'service-name-' + uuid.uuid4().hex}
    endpoint_info.update(attrs)
    endpoint = fakes.FakeResource(info=copy.deepcopy(endpoint_info), loaded=True)
    return endpoint