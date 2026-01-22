from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_post_with_kwargs_only(self):

    class RootController(RestController):

        @expose()
        def get_all(self):
            return 'INDEX'

        @expose('json')
        def post(self, **kw):
            return kw
    app = TestApp(make_app(RootController()))
    r = app.get('/')
    assert r.status_int == 200
    assert r.body == b'INDEX'
    kwargs = {'foo': 'bar', 'spam': 'eggs'}
    r = app.post('/', kwargs)
    assert r.status_int == 200
    assert r.namespace['foo'] == 'bar'
    assert r.namespace['spam'] == 'eggs'