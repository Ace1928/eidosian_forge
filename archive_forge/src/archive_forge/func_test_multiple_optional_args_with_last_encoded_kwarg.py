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
def test_multiple_optional_args_with_last_encoded_kwarg(self):
    r = self.app_.get('/multiple_optional?three=Three%21')
    assert r.status_int == 200
    assert r.body == b'multiple_optional: None, None, Three!'