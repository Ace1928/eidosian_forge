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
def test_list_head_idps(self, iterations=5):
    """List all available IdentityProviders.

        This test collects ids of created IdPs and
        intersects it with the list of all available IdPs.
        List of all IdPs can be a superset of IdPs created in this test,
        because other tests also create IdPs.

        """

    def get_id(resp):
        r = self._fetch_attribute_from_response(resp, 'identity_provider')
        return r.get('id')
    ids = []
    for _ in range(iterations):
        id = get_id(self._create_default_idp())
        ids.append(id)
    ids = set(ids)
    keys_to_check = self.idp_keys
    keys_to_check.append('domain_id')
    url = self.base_url()
    resp = self.get(url)
    self.assertValidListResponse(resp, 'identity_providers', dummy_validator, keys_to_check=keys_to_check)
    entities = self._fetch_attribute_from_response(resp, 'identity_providers')
    entities_ids = set([e['id'] for e in entities])
    ids_intersection = entities_ids.intersection(ids)
    self.assertEqual(ids_intersection, ids)
    self.head(url, expected_status=http.client.OK)