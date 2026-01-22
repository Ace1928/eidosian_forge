from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
def test_extract_member_target_data_inferred(self):
    self.restful_api_resource.member_key = 'argument'
    member_from_driver = self._driver_simulation_get_method
    self.restful_api_resource.get_member_from_driver = member_from_driver
    argument_id = uuid.uuid4().hex
    with self.test_client() as c:
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, argument_id))
        extracted = self.enforcer._extract_member_target_data(member_target_type=None, member_target=None)
        self.assertDictEqual(extracted['target'], self.restful_api_resource().get(argument_id))