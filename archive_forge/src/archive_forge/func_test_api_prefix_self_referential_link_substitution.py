import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
def test_api_prefix_self_referential_link_substitution(self):
    view_arg = uuid.uuid4().hex

    class TestResource(flask_common.ResourceBase):
        api_prefix = '/<string:test_value>/nothing'
    with self.test_request_context(path='/%s/nothing/values' % view_arg, base_url='https://localhost/'):
        flask.request.view_args = {'test_value': view_arg}
        ref = {'id': uuid.uuid4().hex}
        TestResource._add_self_referential_link(ref, collection_name='values')
        self.assertTrue(ref['links']['self'].startswith('https://localhost/v3/%s' % view_arg))