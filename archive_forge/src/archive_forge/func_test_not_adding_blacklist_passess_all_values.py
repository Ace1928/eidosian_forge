import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_not_adding_blacklist_passess_all_values(self):
    """Test a mapping without blacklist specified.

        Not adding a ``blacklist`` keyword to the mapping rules has the same
        effect as adding an empty ``blacklist``. In both cases all values will
        be accepted and passed.

        This test checks scenario where an blacklist was not specified.
        Expected result is to allow any value.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Create group ``NO_EXISTS``
         - Set mapping rules for existing IdP with a blacklist
           that passes through as REMOTE_USER_GROUPS
         - Issue unscoped token with on groups ``EXISTS`` and ``NO_EXISTS``
           assigned

        """
    domain_id = self.domainA['id']
    domain_name = self.domainA['name']
    group_exists = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
    group_exists = PROVIDERS.identity_api.create_group(group_exists)
    group_no_exists = unit.new_group_ref(domain_id=domain_id, name='NO_EXISTS')
    group_no_exists = PROVIDERS.identity_api.create_group(group_no_exists)
    group_ids = set([group_exists['id'], group_no_exists['id']])
    rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS'}]}]}
    PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
    r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
    assigned_group_ids = r.federated_groups
    self.assertEqual(len(group_ids), len(assigned_group_ids))
    for group in assigned_group_ids:
        self.assertIn(group['id'], group_ids)