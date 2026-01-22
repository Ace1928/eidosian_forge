import os
import ssl
from unittest import mock
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
def test_wrap_ssl_version(self):
    self.conf.set_default('ca_file', self.ca_file_name, group=sslutils.config_section)
    self.conf.set_default('version', 'tlsv1', group=sslutils.config_section)
    ssl_kwargs = {'ca_certs': self.conf.ssl.ca_file, 'cert_reqs': ssl.CERT_REQUIRED, 'ssl_version': ssl.PROTOCOL_TLSv1}
    self._test_wrap(**ssl_kwargs)