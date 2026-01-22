import http.client as http_client
import fixtures
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log import log as logging
from requests_mock.contrib import fixture as rm_fixture
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.tests.unit import utils
def create_middleware(self, cb, conf=None, use_global_conf=False):

    @webob.dec.wsgify
    def _do_cb(req):
        return cb(req)
    if use_global_conf:
        opts = conf or {}
    else:
        opts = {'oslo_config_config': self.cfg.conf}
        opts.update(conf or {})
    return auth_token.AuthProtocol(_do_cb, opts)