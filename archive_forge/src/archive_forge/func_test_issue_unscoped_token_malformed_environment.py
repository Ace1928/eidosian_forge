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
def test_issue_unscoped_token_malformed_environment(self):
    """Test whether non string objects are filtered out.

        Put non string objects into the environment, inject
        correct assertion and try to get an unscoped token.
        Expect server not to fail on using split() method on
        non string objects and return token id in the HTTP header.

        """
    environ = {'malformed_object': object(), 'another_bad_idea': tuple(range(10)), 'yet_another_bad_param': dict(zip(uuid.uuid4().hex, range(32)))}
    environ.update(mapping_fixtures.EMPLOYEE_ASSERTION)
    with self.make_request(environ=environ):
        authentication.authenticate_for_token(self.UNSCOPED_V3_SAML2_REQ)