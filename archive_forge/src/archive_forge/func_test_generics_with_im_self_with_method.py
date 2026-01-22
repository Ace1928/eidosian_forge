import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
def test_generics_with_im_self_with_method(self):
    uniq = str(time.time())
    with mock.patch('threading.local', side_effect=AssertionError()):
        app = TestApp(Pecan(self.root(uniq), use_context_locals=False))
        r = app.post_json('/', {'foo': 'bar'}, headers={'X-Unique': uniq})
        assert r.status_int == 200
        json_resp = loads(r.body.decode())
        assert json_resp['foo'] == 'bar'