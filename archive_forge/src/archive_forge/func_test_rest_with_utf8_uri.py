from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
@unittest.skipIf(sys.maxunicode <= 65536, 'narrow python build with UCS-2')
def test_rest_with_utf8_uri(self):

    class FooController(RestController):
        key = chr(127792)
        data = {key: 'Success!'}

        @expose()
        def get_one(self, id_):
            return self.data[id_]

        @expose()
        def get_all(self):
            return 'Hello, World!'

        @expose()
        def put(self, id_, value):
            return self.data[id_]

        @expose()
        def delete(self, id_):
            return self.data[id_]

    class RootController(RestController):
        foo = FooController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foo/%F0%9F%8C%B0')
    assert r.status_int == 200
    assert r.body == b'Success!'
    r = app.put('/foo/%F0%9F%8C%B0', {'value': 'pecans'})
    assert r.status_int == 200
    assert r.body == b'Success!'
    r = app.delete('/foo/%F0%9F%8C%B0')
    assert r.status_int == 200
    assert r.body == b'Success!'
    r = app.get('/foo/')
    assert r.status_int == 200
    assert r.body == b'Hello, World!'