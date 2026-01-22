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
def create_one_credential(attrs=None):
    """Create a fake credential.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with id, type, and so on
        """
    attrs = attrs or {}
    credential_info = {'id': 'credential-id-' + uuid.uuid4().hex, 'type': 'cert', 'user_id': 'user-id-' + uuid.uuid4().hex, 'blob': 'credential-data-' + uuid.uuid4().hex, 'project_id': 'project-id-' + uuid.uuid4().hex, 'links': 'links-' + uuid.uuid4().hex}
    credential_info.update(attrs)
    credential = fakes.FakeResource(info=copy.deepcopy(credential_info), loaded=True)
    return credential