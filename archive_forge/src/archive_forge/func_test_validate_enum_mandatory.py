import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_validate_enum_mandatory(self):

    class Version(object):
        number = wsme.types.wsattr(wsme.types.Enum(str, 'v1', 'v2'), mandatory=True)

    class MyWS(WSRoot):

        @expose(str)
        @validate(Version)
        def setcplx(self, version):
            pass
    r = MyWS(['restjson'])
    app = webtest.TestApp(r.wsgiapp())
    res = app.post_json('/setcplx', params={'version': {}}, expect_errors=True, headers={'Accept': 'application/json'})
    self.assertEqual(res.status_int, 400)