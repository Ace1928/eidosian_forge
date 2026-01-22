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
def test_variable_mixed(self):
    r = self.app_.get('/variable_all/3?month=1&day=12')
    assert r.status_int == 200
    assert r.body == b'variable_all: 3, day=12, month=1'