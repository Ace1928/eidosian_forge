import os
import ssl
from unittest import mock
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
@mock.patch('%s.RuntimeError' % RuntimeError.__module__)
@mock.patch('os.path.exists')
def test_is_enabled(self, exists_mock, runtime_error_mock):
    exists_mock.return_value = True
    self.conf.set_default('cert_file', self.cert_file_name, group=sslutils.config_section)
    self.conf.set_default('key_file', self.key_file_name, group=sslutils.config_section)
    self.conf.set_default('ca_file', self.ca_file_name, group=sslutils.config_section)
    sslutils.is_enabled(self.conf)
    self.assertFalse(runtime_error_mock.called)