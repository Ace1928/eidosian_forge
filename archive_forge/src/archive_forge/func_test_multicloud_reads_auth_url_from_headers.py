from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
@mock.patch.object(auth_url.AuthUrlFilter, '_validate_auth_url')
@mock.patch.object(auth_url.cfg, 'CONF')
def test_multicloud_reads_auth_url_from_headers(self, mock_cfg, mock_val):
    mock_cfg.auth_password.multi_cloud = True
    mock_val.return_value = True
    req = webob.Request.blank('/tenant_id/')
    req.headers['X-Auth-Url'] = 'overwrites config'
    self.middleware(req)
    self.assertIn('X-Auth-Url', req.headers)
    self.assertEqual('overwrites config', req.headers['X-Auth-Url'])