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
def create_one_endpointgroup_filter(attrs=None):
    """Create a fake endpoint project relationship.

        :param Dictionary attrs:
            A dictionary with all attributes of endpointgroup filter
        :return:
            A FakeResource object with project, endpointgroup and so on
        """
    attrs = attrs or {}
    endpointgroup_filter_info = {'project': 'project-id-' + uuid.uuid4().hex, 'endpointgroup': 'endpointgroup-id-' + uuid.uuid4().hex}
    endpointgroup_filter_info.update(attrs)
    endpointgroup_filter = fakes.FakeModel(copy.deepcopy(endpointgroup_filter_info))
    return endpointgroup_filter