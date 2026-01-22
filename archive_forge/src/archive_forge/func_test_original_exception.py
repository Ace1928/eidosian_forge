import json
from webtest import TestApp
import pecan
from pecan.middleware.errordocument import ErrorDocumentMiddleware
from pecan.middleware.recursive import RecursiveMiddleware
from pecan.tests import PecanTestCase
def test_original_exception(self):

    class RootController(object):

        @pecan.expose()
        def index(self):
            if pecan.request.method != 'POST':
                pecan.abort(405, 'You have to POST, dummy!')
            return 'Hello, World!'

        @pecan.expose('json')
        def error(self, status):
            return dict(status=int(status), reason=pecan.request.environ['pecan.original_exception'].detail)
    app = pecan.Pecan(RootController())
    app = RecursiveMiddleware(ErrorDocumentMiddleware(app, {405: '/error/405'}))
    app = TestApp(app)
    assert app.post('/').status_int == 200
    r = app.get('/', expect_errors=405)
    assert r.status_int == 405
    resp = json.loads(r.body.decode())
    assert resp['status'] == 405
    assert resp['reason'] == 'You have to POST, dummy!'