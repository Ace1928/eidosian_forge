from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_rest_with_utf8_endpoint(self):

    class ChildController(object):

        @expose()
        def index(self):
            return 'Hello, World!'

    class FooController(RestController):
        pass
    setattr(FooController, '🌰', ChildController())

    class RootController(RestController):
        foo = FooController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foo/%F0%9F%8C%B0/')
    assert r.status_int == 200
    assert r.body == b'Hello, World!'