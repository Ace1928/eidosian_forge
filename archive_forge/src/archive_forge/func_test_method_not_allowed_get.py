from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_method_not_allowed_get(self):

    class ThingsController(RestController):

        @expose()
        def put(self, id_, value):
            response.status = 200

        @expose()
        def delete(self, id_):
            response.status = 200
    app = TestApp(make_app(ThingsController()))
    r = app.get('/', status=405)
    assert r.status_int == 405
    assert r.headers['Allow'] == 'DELETE, PUT'