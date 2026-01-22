import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_multiple_expose(self):

    class MyRoot(WSRoot):

        def multiply(self, a, b):
            return a * b
        mul_int = expose(int, int, int, wrap=True)(multiply)
        mul_float = expose(float, float, float, wrap=True)(multiply)
        mul_string = expose(wsme.types.text, wsme.types.text, int, wrap=True)(multiply)
    r = MyRoot(['restjson'])
    app = webtest.TestApp(r.wsgiapp())
    res = app.get('/mul_int?a=2&b=5', headers={'Accept': 'application/json'})
    self.assertEqual(res.body, b'10')
    res = app.get('/mul_float?a=1.2&b=2.9', headers={'Accept': 'application/json'})
    self.assertEqual(res.body, b'3.48')
    res = app.get('/mul_string?a=hello&b=2', headers={'Accept': 'application/json'})
    self.assertEqual(res.body, b'"hellohello"')