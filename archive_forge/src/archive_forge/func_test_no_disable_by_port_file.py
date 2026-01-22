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
def test_no_disable_by_port_file(self):
    filename = self.create_tempfiles([('test', 'foobar')])[0]
    conf = {'backends': 'disable_by_files_ports', 'disable_by_file_paths': '8000:%s' % filename}
    self._do_test(conf, expected_code=webob.exc.HTTPOk.code, expected_body=b'OK')
    self.assertIn('disable_by_files_ports', self.app._backends.names())