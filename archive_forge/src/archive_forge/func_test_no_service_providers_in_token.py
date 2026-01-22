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
def test_no_service_providers_in_token(self):
    """Test service catalog with disabled service providers.

        There should be no entry ``service_providers`` in the catalog.
        Test passes providing no attribute was raised.

        """
    sp_ref = {'enabled': False}
    for sp in (self.SP1, self.SP2, self.SP3):
        PROVIDERS.federation_api.update_sp(sp, sp_ref)
    model = token_model.TokenModel()
    model.user_id = self.user_id
    model.methods = ['password']
    token = render_token.render_token_response_from_model(model)
    self.assertNotIn('service_providers', token['token'], message='Expected Service Catalog not to have service_providers')