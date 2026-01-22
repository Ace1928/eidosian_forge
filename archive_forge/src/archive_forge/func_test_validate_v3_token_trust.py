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
def test_validate_v3_token_trust(self):
    domain_ref = unit.new_domain_ref()
    domain_ref = PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
    user_ref = unit.new_user_ref(domain_ref['id'])
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    trustor_user_ref = unit.new_user_ref(domain_ref['id'])
    trustor_user_ref = PROVIDERS.identity_api.create_user(trustor_user_ref)
    project_ref = unit.new_project_ref(domain_id=domain_ref['id'])
    project_ref = PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
    role_ref = unit.new_role_ref()
    role_ref = PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=user_ref['id'], project_id=project_ref['id'])
    PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=trustor_user_ref['id'], project_id=project_ref['id'])
    trustor_user_id = trustor_user_ref['id']
    trustee_user_id = user_ref['id']
    trust_ref = unit.new_trust_ref(trustor_user_id, trustee_user_id, project_id=project_ref['id'], role_ids=[role_ref['id']])
    trust_ref = PROVIDERS.trust_api.create_trust(trust_ref['id'], trust_ref, trust_ref['roles'])
    method_names = ['password']
    token = PROVIDERS.token_provider_api.issue_token(user_ref['id'], method_names, project_id=project_ref['id'], trust_id=trust_ref['id'])
    token = PROVIDERS.token_provider_api.validate_token(token.id)
    self.assertEqual(trust_ref['id'], token.trust_id)
    self.assertFalse(token.trust['impersonation'])
    self.assertEqual(user_ref['id'], token.trustee['id'])
    self.assertEqual(trustor_user_ref['id'], token.trustor['id'])