from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_getall_with_trailing_slash(self):

    class ThingsController(RestController):
        data = ['zero', 'one', 'two', 'three']

        @expose('json')
        def get_all(self):
            return dict(items=self.data)

    class RootController(object):
        things = ThingsController()
    app = TestApp(make_app(RootController()))
    r = app.get('/things/')
    assert r.status_int == 200
    assert r.body == dumps(dict(items=ThingsController.data)).encode('utf-8')