import json
from webtest import TestApp
import pecan
from pecan.middleware.errordocument import ErrorDocumentMiddleware
from pecan.middleware.recursive import RecursiveMiddleware
from pecan.tests import PecanTestCase
def test_middleware_routes_to_404_message(self):
    r = self.app.get('/', expect_errors=True)
    assert r.status_int == 404
    assert r.body == b'Error: 404'