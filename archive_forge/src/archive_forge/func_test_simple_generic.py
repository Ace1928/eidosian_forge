from json import dumps
from webtest import TestApp
from pecan import Pecan, expose, abort
from pecan.tests import PecanTestCase
def test_simple_generic(self):

    class RootController(object):

        @expose(generic=True)
        def index(self):
            pass

        @index.when(method='POST', template='json')
        def do_post(self):
            return dict(result='POST')

        @index.when(method='GET')
        def do_get(self):
            return 'GET'
    app = TestApp(Pecan(RootController()))
    r = app.get('/')
    assert r.status_int == 200
    assert r.body == b'GET'
    r = app.post('/')
    assert r.status_int == 200
    assert r.body == dumps(dict(result='POST')).encode('utf-8')
    r = app.get('/do_get', status=404)
    assert r.status_int == 404