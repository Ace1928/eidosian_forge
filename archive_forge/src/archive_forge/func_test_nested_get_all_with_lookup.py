from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_nested_get_all_with_lookup(self):

    class BarsController(RestController):

        @expose()
        def get_one(self, foo_id, id):
            return '4'

        @expose()
        def get_all(self, foo_id):
            return '3'

        @expose('json')
        def _lookup(self, id, *remainder):
            redirect('/lookup-hit/')

    class FoosController(RestController):
        bars = BarsController()

        @expose()
        def get_one(self, id):
            return '2'

        @expose()
        def get_all(self):
            return '1'

    class RootController(object):
        foos = FoosController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foos/')
    assert r.status_int == 200
    assert r.body == b'1'
    r = app.get('/foos/1/')
    assert r.status_int == 200
    assert r.body == b'2'
    r = app.get('/foos/1/bars/')
    assert r.status_int == 200
    assert r.body == b'3'
    r = app.get('/foos/1/bars/2/')
    assert r.status_int == 200
    assert r.body == b'4'
    r = app.get('/foos/bars/')
    assert r.status_int == 302
    assert r.headers['Location'].endswith('/lookup-hit/')
    r = app.get('/foos/bars/1')
    assert r.status_int == 302
    assert r.headers['Location'].endswith('/lookup-hit/')