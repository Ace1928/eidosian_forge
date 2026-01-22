import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
import fixtures
from oslo_log import log
from oslo_utils import timeutils
from keystone import auth
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider
from keystone.token.providers import fernet
from keystone.token import token_formatters
def test_federated_project_scoped_payload(self):
    exp_federated_group_ids = [{'id': 'someNonUuidGroupId'}]
    exp_idp_id = uuid.uuid4().hex
    exp_protocol_id = uuid.uuid4().hex
    self._test_payload(token_formatters.FederatedProjectScopedPayload, exp_user_id='someNonUuidUserId', exp_methods=['token'], exp_project_id=uuid.uuid4().hex, exp_federated_group_ids=exp_federated_group_ids, exp_identity_provider_id=exp_idp_id, exp_protocol_id=exp_protocol_id)