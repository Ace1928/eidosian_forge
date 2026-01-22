import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_handle_request(self):

    class MyRoot(WSRoot):

        @expose()
        def touch(self):
            pass
    p = DummyProtocol()
    r = MyRoot(protocols=[p])
    app = webtest.TestApp(r.wsgiapp())
    res = app.get('/')
    assert p.lastreq.path == '/'
    assert p.hits == 1
    res = app.get('/touch?wsmeproto=dummy')
    assert p.lastreq.path == '/touch'
    assert p.hits == 2

    class NoPathProto(DummyProtocol):

        def extract_path(self, request):
            return None
    p = NoPathProto()
    r = MyRoot(protocols=[p])
    app = webtest.TestApp(r.wsgiapp())
    res = app.get('/', expect_errors=True)
    print(res.status, res.body)
    assert res.status_int == 400