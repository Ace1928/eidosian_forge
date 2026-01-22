import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_update_ephemeral_user(self):
    federated_user_a = model.FederatedUser()
    federated_user_b = model.FederatedUser()
    federated_user_a.idp_id = 'a_idp'
    federated_user_b.idp_id = 'b_idp'
    federated_user_a.display_name = 'federated_a'
    federated_user_b.display_name = 'federated_b'
    federated_users = [federated_user_a, federated_user_b]
    user_a = model.User()
    user_a.federated_users = federated_users
    self.assertEqual(federated_user_a.display_name, user_a.name)
    self.assertIsNone(user_a.password)
    user_a.name = 'new_federated_a'
    self.assertEqual('new_federated_a', user_a.name)
    self.assertIsNone(user_a.local_user)