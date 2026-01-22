import threading
import time
from unittest import mock
from oslo_config import fixture as config
from oslo_serialization import jsonutils
from oslotest import base as test_base
import requests
import webob.dec
import webob.exc
from oslo_middleware import healthcheck
from oslo_middleware.healthcheck import __main__
@mock.patch('oslo_middleware.healthcheck.disable_by_file.LOG')
def test_disablefile_unconfigured(self, fake_log):
    fake_warn = fake_log.warning
    conf = {'backends': 'disable_by_file'}
    self._do_test(conf, expected_body=b'OK')
    self.assertIn('disable_by_file', self.app._backends.names())
    fake_warn.assert_called_once_with('DisableByFile healthcheck middleware enabled without disable_by_file_path set')