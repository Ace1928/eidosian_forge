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
def create_one_group(attrs=None):
    """Create a fake group.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, name, and so on
        """
    attrs = attrs or {}
    group_info = {'id': 'group-id-' + uuid.uuid4().hex, 'name': 'group-name-' + uuid.uuid4().hex, 'links': 'links-' + uuid.uuid4().hex, 'domain_id': 'domain-id-' + uuid.uuid4().hex, 'description': 'group-description-' + uuid.uuid4().hex}
    group_info.update(attrs)
    group = fakes.FakeResource(info=copy.deepcopy(group_info), loaded=True)
    return group