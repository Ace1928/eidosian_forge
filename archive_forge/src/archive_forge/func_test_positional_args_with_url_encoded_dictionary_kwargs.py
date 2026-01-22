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
def test_positional_args_with_url_encoded_dictionary_kwargs(self):
    r = self.app_.post('/multiple', {'one': 'Five%20', 'two': 'Six%20%21'})
    assert r.status_int == 200
    assert r.body == b'multiple: Five%20, Six%20%21'