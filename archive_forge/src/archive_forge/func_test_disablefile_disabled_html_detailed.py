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
def test_disablefile_disabled_html_detailed(self):
    filename = self.create_tempfiles([('test', 'foobar')])[0]
    conf = {'backends': 'disable_by_file', 'disable_by_file_path': filename, 'detailed': True}
    res = self._do_test_request(conf, accept='text/html')
    self.assertIn(b'<TD>DISABLED BY FILE</TD>', res.body)
    self.assertEqual(webob.exc.HTTPServiceUnavailable.code, res.status_int)