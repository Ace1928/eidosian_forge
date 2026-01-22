from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_basic_rest(self):

    class OthersController(object):

        @expose()
        def index(self):
            return 'OTHERS'

        @expose()
        def echo(self, value):
            return str(value)

    class ThingsController(RestController):
        data = ['zero', 'one', 'two', 'three']
        _custom_actions = {'count': ['GET'], 'length': ['GET', 'POST']}
        others = OthersController()

        @expose()
        def get_one(self, id):
            return self.data[int(id)]

        @expose('json')
        def get_all(self):
            return dict(items=self.data)

        @expose()
        def length(self, id, value=None):
            length = len(self.data[int(id)])
            if value:
                length += len(value)
            return str(length)

        @expose()
        def get_count(self):
            return str(len(self.data))

        @expose()
        def new(self):
            return 'NEW'

        @expose()
        def post(self, value):
            self.data.append(value)
            response.status = 302
            return 'CREATED'

        @expose()
        def edit(self, id):
            return 'EDIT %s' % self.data[int(id)]

        @expose()
        def put(self, id, value):
            self.data[int(id)] = value
            return 'UPDATED'

        @expose()
        def get_delete(self, id):
            return 'DELETE %s' % self.data[int(id)]

        @expose()
        def delete(self, id):
            del self.data[int(id)]
            return 'DELETED'

        @expose()
        def trace(self):
            return 'TRACE'

        @expose()
        def post_options(self):
            return 'OPTIONS'

        @expose()
        def options(self):
            abort(500)

        @expose()
        def other(self):
            abort(500)

    class RootController(object):
        things = ThingsController()
    app = TestApp(make_app(RootController()))
    r = app.get('/things')
    assert r.status_int == 200
    assert r.body == dumps(dict(items=ThingsController.data)).encode('utf-8')
    for i, value in enumerate(ThingsController.data):
        r = app.get('/things/%d' % i)
        assert r.status_int == 200
        assert r.body == value.encode('utf-8')
    r = app.post('/things', {'value': 'four'})
    assert r.status_int == 302
    assert r.body == b'CREATED'
    r = app.get('/things/4')
    assert r.status_int == 200
    assert r.body == b'four'
    r = app.get('/things/3/edit')
    assert r.status_int == 200
    assert r.body == b'EDIT three'
    r = app.put('/things/4', {'value': 'FOUR'})
    assert r.status_int == 200
    assert r.body == b'UPDATED'
    r = app.get('/things/4')
    assert r.status_int == 200
    assert r.body == b'FOUR'
    r = app.get('/things/4?_method=put', {'value': 'FOUR!'}, status=405)
    assert r.status_int == 405
    r = app.get('/things/4')
    assert r.status_int == 200
    assert r.body == b'FOUR'
    r = app.post('/things/4?_method=put', {'value': 'FOUR!'})
    assert r.status_int == 200
    assert r.body == b'UPDATED'
    r = app.get('/things/4')
    assert r.status_int == 200
    assert r.body == b'FOUR!'
    r = app.get('/things/4/delete')
    assert r.status_int == 200
    assert r.body == b'DELETE FOUR!'
    r = app.delete('/things/4')
    assert r.status_int == 200
    assert r.body == b'DELETED'
    r = app.get('/things')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 4
    r = app.get('/things/3?_method=DELETE', status=405)
    assert r.status_int == 405
    r = app.get('/things')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 4
    r = app.post('/things/3?_method=DELETE')
    assert r.status_int == 200
    assert r.body == b'DELETED'
    r = app.get('/things')
    assert r.status_int == 200
    assert len(loads(r.body.decode())['items']) == 3
    r = app.request('/things', method='TRACE')
    assert r.status_int == 200
    assert r.body == b'TRACE'
    r = app.get('/things?_method=TRACE')
    assert r.status_int == 200
    assert r.body == b'TRACE'
    r = app.request('/things', method='OPTIONS')
    assert r.status_int == 200
    assert r.body == b'OPTIONS'
    r = app.post('/things', {'_method': 'OPTIONS'})
    assert r.status_int == 200
    assert r.body == b'OPTIONS'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = app.request('/things/other', method='MISC', status=405)
        assert r.status_int == 405
    r = app.post('/things/other', {'_method': 'MISC'}, status=405)
    assert r.status_int == 405
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = app.request('/things/others/', method='MISC')
        assert r.status_int == 200
        assert r.body == b'OTHERS'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = app.request('/things/others', method='MISC', status=302)
        assert r.status_int == 302
    r = app.get('/things/others/?_method=MISC')
    assert r.status_int == 200
    assert r.body == b'OTHERS'
    r = app.get('/things?_method=BAD', status=405)
    assert r.status_int == 405
    r = app.get('/things/count')
    assert r.status_int == 200
    assert r.body == b'3'
    r = app.get('/things/1/length')
    assert r.status_int == 200
    assert r.body == b'3'
    r = app.get('/things/others/echo?value=test')
    assert r.status_int == 200
    assert r.body == b'test'
    r = app.post('/things/1/length', {'value': 'test'})
    assert r.status_int == 200
    assert r.body == b'7'
    r = app.post('/things/others/echo', {'value': 'test'})
    assert r.status_int == 200
    assert r.body == b'test'