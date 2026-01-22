import os
import ssl
from unittest import mock
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
@mock.patch('os.path.exists')
def test_is_enabled_no_ssl_ca_file_fails(self, exists_mock):
    exists_mock.side_effect = [True, True, False]
    self.conf.set_default('cert_file', self.cert_file_name, group=sslutils.config_section)
    self.conf.set_default('key_file', self.key_file_name, group=sslutils.config_section)
    self.conf.set_default('ca_file', '/no/such/file', group=sslutils.config_section)
    self.assertRaises(RuntimeError, sslutils.is_enabled, self.conf)