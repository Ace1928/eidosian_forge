from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_nested_rest_with_missing_intermediate_id(self):

    class BarsController(RestController):
        data = [['zero-zero', 'zero-one'], ['one-zero', 'one-one']]

        @expose('json')
        def get_all(self, foo_id):
            return dict(items=self.data[int(foo_id)])

    class FoosController(RestController):
        data = ['zero', 'one']
        bars = BarsController()

        @expose()
        def get_one(self, id):
            return self.data[int(id)]

        @expose('json')
        def get_all(self):
            return dict(items=self.data)

    class RootController(object):
        foos = FoosController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foos')
    self.assertEqual(r.status_int, 200)
    self.assertEqual(r.body, dumps(dict(items=FoosController.data)).encode('utf-8'))
    r = app.get('/foos/1/bars')
    self.assertEqual(r.status_int, 200)
    self.assertEqual(r.body, dumps(dict(items=BarsController.data[1])).encode('utf-8'))
    r = app.get('/foos/bars', expect_errors=True)
    self.assertEqual(r.status_int, 404)