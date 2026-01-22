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
def test_valid_saml_xml(self):
    """Test the generated SAML object can become valid XML.

        Test the generator directly by passing known arguments, the result
        should be a SAML object that consistently includes attributes based on
        the known arguments that were passed in.

        """
    with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
        generator = keystone_idp.SAMLGenerator()
        response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
    saml_str = response.to_string()
    response = etree.fromstring(saml_str)
    issuer = response[0]
    assertion = response[2]
    self.assertEqual(self.RECIPIENT, response.get('Destination'))
    self.assertEqual(self.ISSUER, issuer.text)
    user_attribute = assertion[4][0]
    self.assertEqual(self.SUBJECT, user_attribute[0].text)
    user_domain_attribute = assertion[4][1]
    self.assertEqual(self.SUBJECT_DOMAIN, user_domain_attribute[0].text)
    role_attribute = assertion[4][2]
    for attribute_value in role_attribute:
        self.assertIn(attribute_value.text, self.ROLES)
    project_attribute = assertion[4][3]
    self.assertEqual(self.PROJECT, project_attribute[0].text)
    project_domain_attribute = assertion[4][4]
    self.assertEqual(self.PROJECT_DOMAIN, project_domain_attribute[0].text)
    group_attribute = assertion[4][5]
    for attribute_value in group_attribute:
        self.assertIn(attribute_value.text, self.GROUPS)