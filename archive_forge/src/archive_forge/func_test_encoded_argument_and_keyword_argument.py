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
def test_encoded_argument_and_keyword_argument(self):
    r = self.app_.get('/This%20is%20a%20test%21?id=three')
    assert r.status_int == 200
    assert r.body == b'index: This is a test!'