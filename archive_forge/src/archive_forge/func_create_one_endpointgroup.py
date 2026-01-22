import copy
import datetime
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from osc_lib.cli import format_columns
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_one_endpointgroup(attrs=None):
    """Create a fake endpoint group.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, url, and so on
        """
    attrs = attrs or {}
    endpointgroup_info = {'id': 'endpoint-group-id-' + uuid.uuid4().hex, 'name': 'endpoint-group-name-' + uuid.uuid4().hex, 'filters': {'region': 'region-' + uuid.uuid4().hex, 'service_id': 'service-id-' + uuid.uuid4().hex}, 'description': 'endpoint-group-description-' + uuid.uuid4().hex, 'links': 'links-' + uuid.uuid4().hex}
    endpointgroup_info.update(attrs)
    endpoint = fakes.FakeResource(info=copy.deepcopy(endpointgroup_info), loaded=True)
    return endpoint