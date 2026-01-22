import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_return_content_type_guess(self):

    class DummierProto(DummyProtocol):
        content_types = ['text/xml', 'text/plain']
    r = WSRoot([DummierProto()])
    app = webtest.TestApp(r.wsgiapp())
    res = app.get('/', expect_errors=True, headers={'Accept': 'text/xml,q=0.8'})
    assert res.status_int == 400
    assert res.content_type == 'text/xml', res.content_type
    res = app.get('/', expect_errors=True, headers={'Accept': 'text/plain'})
    assert res.status_int == 400
    assert res.content_type == 'text/plain', res.content_type