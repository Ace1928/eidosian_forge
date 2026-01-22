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
def test_source_within_allowed_ranges(self):
    conf = {'allowed_source_ranges': ['192.168.0.0/24', '192.168.1.0/24']}
    self._do_test(conf, expected_code=webob.exc.HTTPOk.code, remote_addr='192.168.0.1')