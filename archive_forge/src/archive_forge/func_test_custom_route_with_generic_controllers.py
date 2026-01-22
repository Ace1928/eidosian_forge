import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
def test_custom_route_with_generic_controllers(self):

    class RootController(object):

        @expose(route='some-path', generic=True)
        def foo(self):
            return 'Hello, World!'

        @foo.when(method='POST')
        def handle_post(self):
            return 'POST!'
    app = TestApp(Pecan(RootController()))
    r = app.get('/some-path/')
    assert r.status_int == 200
    assert r.body == b'Hello, World!'
    r = app.get('/foo/', expect_errors=True)
    assert r.status_int == 404
    r = app.post('/some-path/')
    assert r.status_int == 200
    assert r.body == b'POST!'
    r = app.post('/foo/', expect_errors=True)
    assert r.status_int == 404