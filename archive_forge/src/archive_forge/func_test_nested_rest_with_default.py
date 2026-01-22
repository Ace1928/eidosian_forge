from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_nested_rest_with_default(self):

    class FooController(RestController):

        @expose()
        def _default(self, *remainder):
            return 'DEFAULT %s' % remainder

    class RootController(RestController):
        foo = FooController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foo/missing')
    assert r.status_int == 200
    assert r.body == b'DEFAULT missing'