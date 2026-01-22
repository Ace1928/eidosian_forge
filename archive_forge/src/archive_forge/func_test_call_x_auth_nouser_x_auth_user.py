import json
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
import requests
from heat.api.aws import ec2token
from heat.api.aws import exception
from heat.common import wsgi
from heat.tests import common
from heat.tests import utils
def test_call_x_auth_nouser_x_auth_user(self):
    req_env = {'HTTP_X_AUTH_USER': 'foo', 'HTTP_AUTHORIZATION': 'Authorization: foo SignedHeaders=content-type;host;x-amz-date,Signature=xyz'}
    dummy_req = self._dummy_GET_request(environ=req_env)
    ec2 = ec2token.EC2Token(app='xyz', conf={})
    self.assertEqual('xyz', ec2.__call__(dummy_req))