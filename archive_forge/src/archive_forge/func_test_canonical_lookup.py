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
def test_canonical_lookup(self):
    assert self.app_.get('/users', expect_errors=404).status_int == 404
    assert self.app_.get('/users/', expect_errors=404).status_int == 404
    assert self.app_.get('/users/100').status_int == 302
    assert self.app_.get('/users/100/').body == b'100'