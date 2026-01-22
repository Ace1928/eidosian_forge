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
def test_correct_json_home_document(self):

    class MappedResource(flask_restful.Resource):

        def post(self):
            rbac_enforcer.enforcer.RBACEnforcer().enforce_call(action='example:allowed')
            post_body = flask.request.get_json()
            return {'post_body': post_body}
    json_home_data = {'https://docs.openstack.org/api/openstack-identity/3/rel/argument': {'href-template': '/v3/arguments/{argument_id}', 'href-vars': {'argument_id': 'https://docs.openstack.org/api/openstack-identity/3/param/argument_id'}}, 'https://docs.openstack.org/api/openstack-identity/3/rel/arguments': {'href': '/v3/arguments'}, 'https://docs.openstack.org/api/openstack-identity/3/rel/test': {'href': '/v3/test_api'}}
    resource_map = flask_common.construct_resource_map(resource=MappedResource, url='test_api', alternate_urls=[], resource_kwargs={}, rel='test', status=json_home.Status.STABLE, path_vars=None, resource_relation_func=json_home.build_v3_resource_relation)
    restful_api = _TestRestfulAPI(resource_mapping=[resource_map])
    self.public_app.app.register_blueprint(restful_api.blueprint)
    with self.test_client() as c:
        headers = {'Accept': 'application/json-home'}
        resp = c.get('/', headers=headers)
        resp_data = jsonutils.loads(resp.data)
        for rel in json_home_data:
            self.assertThat(resp_data['resources'][rel], matchers.Equals(json_home_data[rel]))