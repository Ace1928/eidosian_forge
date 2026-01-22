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
def test_variable_post_mixed(self):
    r = self.app_.post('/variable_all/7', {'id': 'seven', 'month': '1', 'day': '12'})
    assert r.status_int == 200
    assert r.body == b'variable_all: 7, day=12, id=seven, month=1'