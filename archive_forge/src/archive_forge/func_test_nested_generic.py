from json import dumps
from webtest import TestApp
from pecan import Pecan, expose, abort
from pecan.tests import PecanTestCase
def test_nested_generic(self):

    class SubSubController(object):

        @expose(generic=True)
        def index(self):
            return 'GET'

        @index.when(method='DELETE', template='json')
        def do_delete(self, name, *args):
            return dict(result=name, args=', '.join(args))

    class SubController(object):
        sub = SubSubController()

    class RootController(object):
        sub = SubController()
    app = TestApp(Pecan(RootController()))
    r = app.get('/sub/sub/')
    assert r.status_int == 200
    assert r.body == b'GET'
    r = app.delete('/sub/sub/joe/is/cool')
    assert r.status_int == 200
    assert r.body == dumps(dict(result='joe', args='is, cool')).encode('utf-8')