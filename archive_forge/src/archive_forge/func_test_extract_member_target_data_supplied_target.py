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
def test_extract_member_target_data_supplied_target(self):
    member_type = uuid.uuid4().hex
    member_target = {uuid.uuid4().hex: {uuid.uuid4().hex}}
    extracted = self.enforcer._extract_member_target_data(member_target_type=member_type, member_target=member_target)
    self.assertDictEqual({'target': {member_type: member_target}}, extracted)