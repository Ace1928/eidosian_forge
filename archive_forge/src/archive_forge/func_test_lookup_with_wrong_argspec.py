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
def test_lookup_with_wrong_argspec(self):

    class RootController(object):

        @expose()
        def _lookup(self, someID):
            return 'Bad arg spec'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        app = TestApp(Pecan(RootController(), use_context_locals=False))
        r = app.get('/foo/bar', expect_errors=True)
        assert r.status_int == 404