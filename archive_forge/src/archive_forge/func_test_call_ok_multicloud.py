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
def test_call_ok_multicloud(self):
    dummy_conf = {'allowed_auth_uris': ['http://123:5000/v2.0', 'http://456:5000/v2.0'], 'multi_cloud': True}
    ec2 = ec2token.EC2Token(app='woot', conf=dummy_conf)
    params = {'AWSAccessKeyId': 'foo', 'Signature': 'xyz'}
    req_env = {'SERVER_NAME': 'heat', 'SERVER_PORT': '8000', 'PATH_INFO': '/v1'}
    dummy_req = self._dummy_GET_request(params, req_env)
    ok_resp = json.dumps({'token': {'project': {'name': 'tenant', 'id': 'abcd1234'}}})
    err_msg = 'EC2 access key not found.'
    err_resp = json.dumps({'error': {'message': err_msg}})
    m_p = self._stub_http_connection(req_url='http://123:5000/v2.0/ec2tokens', response=err_resp, params={'AWSAccessKeyId': 'foo'}, direct_mock=False)
    m_p2 = self._stub_http_connection(req_url='http://456:5000/v2.0/ec2tokens', response=ok_resp, params={'AWSAccessKeyId': 'foo'}, direct_mock=False)
    requests.post.side_effect = [m_p, m_p2]
    self.assertEqual('woot', ec2.__call__(dummy_req))
    self.assertEqual(2, requests.post.call_count)
    requests.post.assert_called_with(self.verify_req_url, data=self.verify_data, verify=self.verify_verify, cert=self.verify_cert, headers=self.verify_req_headers)