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
def test_call_err_multicloud_none_allowed(self):
    dummy_conf = {'allowed_auth_uris': [], 'multi_cloud': True}
    ec2 = ec2token.EC2Token(app='woot', conf=dummy_conf)
    params = {'AWSAccessKeyId': 'foo', 'Signature': 'xyz'}
    req_env = {'SERVER_NAME': 'heat', 'SERVER_PORT': '8000', 'PATH_INFO': '/v1'}
    dummy_req = self._dummy_GET_request(params, req_env)
    self.assertRaises(exception.HeatAccessDeniedError, ec2.__call__, dummy_req)