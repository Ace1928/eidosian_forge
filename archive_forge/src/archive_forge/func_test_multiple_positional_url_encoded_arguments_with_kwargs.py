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
def test_multiple_positional_url_encoded_arguments_with_kwargs(self):
    r = self.app_.post('/optional/Some%20Number', {'id': 'five'})
    assert r.status_int == 200
    assert r.body == b'optional: Some Number'