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
def test_required_argument(self):
    try:
        r = self.app_.get('/')
        assert r.status_int != 200
    except Exception as ex:
        assert type(ex) == TypeError
        assert ex.args[0] in ('index() takes exactly 2 arguments (1 given)', "index() missing 1 required positional argument: 'id'", "TestControllerArguments.app_.<locals>.RootController.index() missing 1 required positional argument: 'id'")