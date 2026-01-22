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
def test_threadlocal_argument_warning_on_generic_delegate(self):
    with mock.patch('threading.local', side_effect=AssertionError()):
        app = TestApp(Pecan(self.root(), use_context_locals=False))
        self.assertRaises(TypeError, app.put, '/generic/')