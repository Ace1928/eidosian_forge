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
def test_sign_assertion_logs_message_if_xmlsec1_is_not_installed(self):
    with mock.patch.object(subprocess, 'check_output') as co_mock:
        co_mock.side_effect = subprocess.CalledProcessError(returncode=1, cmd=CONF.saml.xmlsec1_binary)
        logger_fixture = self.useFixture(fixtures.LoggerFixture())
        self.assertRaises(exception.SAMLSigningError, keystone_idp._sign_assertion, self.signed_assertion)
        expected_log = 'Unable to locate xmlsec1 binary on the system. Check to make sure it is installed.\n'
        self.assertIn(expected_log, logger_fixture.output)