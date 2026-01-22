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
def test_json_response(self):
    expected_body = jsonutils.dumps({'detailed': False, 'reasons': []}, indent=4, sort_keys=True).encode('utf-8')
    self._do_test(expected_body=expected_body, accept='application/json')