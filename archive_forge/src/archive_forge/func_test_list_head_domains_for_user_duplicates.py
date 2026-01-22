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
def test_list_head_domains_for_user_duplicates(self):
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    user_id, unscoped_token = self._authenticate_via_saml()
    r = self.get('/OS-FEDERATION/domains', token=unscoped_token)
    group_domains = r.result['domains']
    domain_from_group = group_domains[0]
    self.head('/OS-FEDERATION/domains', token=unscoped_token, expected_status=http.client.OK)
    PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=user_id, domain_id=domain_from_group['id'])
    r = self.get('/OS-FEDERATION/domains', token=unscoped_token)
    user_domains = r.result['domains']
    user_domain_ids = []
    for domain in user_domains:
        self.assertNotIn(domain['id'], user_domain_ids)
        user_domain_ids.append(domain['id'])
    r = self.get('/auth/domains', token=unscoped_token)
    user_domains = r.result['domains']
    user_domain_ids = []
    for domain in user_domains:
        self.assertNotIn(domain['id'], user_domain_ids)
        user_domain_ids.append(domain['id'])