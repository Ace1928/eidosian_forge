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
def test_saml_signing(self):
    """Test that the SAML generator produces a SAML object.

        Test the SAML generator directly by passing known arguments, the result
        should be a SAML object that consistently includes attributes based on
        the known arguments that were passed in.

        """
    if not _is_xmlsec1_installed():
        self.skipTest('xmlsec1 is not installed')
    generator = keystone_idp.SAMLGenerator()
    response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
    signature = response.assertion.signature
    self.assertIsNotNone(signature)
    self.assertIsInstance(signature, xmldsig.Signature)
    idp_public_key = sigver.read_cert_from_file(CONF.saml.certfile, 'pem')
    cert_text = signature.key_info.x509_data[0].x509_certificate.text
    cert_text = cert_text.replace(os.linesep, '')
    self.assertEqual(idp_public_key, cert_text)