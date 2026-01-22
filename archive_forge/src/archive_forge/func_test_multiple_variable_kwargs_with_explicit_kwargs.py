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
def test_multiple_variable_kwargs_with_explicit_kwargs(self):
    r = self.app_.get('/variable_kwargs?id=2&dummy=dummy')
    assert r.status_int == 200
    assert r.body == b'variable_kwargs: dummy=dummy, id=2'