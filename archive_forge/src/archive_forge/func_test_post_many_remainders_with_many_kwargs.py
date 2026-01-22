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
def test_post_many_remainders_with_many_kwargs(self):
    r = self.app_.post('/eater/10', {'id': 'ten', 'month': '1', 'day': '12', 'dummy': 'dummy'})
    assert r.status_int == 200
    assert r.body == b'eater: 10, dummy, day=12, month=1'