from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
def test_ForwardRequest_factory(self):

    class TestForwardRequestMiddleware(Middleware):

        def __call__(self, environ, start_response):
            if environ['PATH_INFO'] != '/not_found':
                return self.app(environ, start_response)
            environ['PATH_INFO'] = self.url

            def factory(app):

                class WSGIApp(object):

                    def __init__(self, app):
                        self.app = app

                    def __call__(self, e, start_response):

                        def keep_status_start_response(status, headers, exc_info=None):
                            return start_response('404 Not Found', headers, exc_info)
                        return self.app(e, keep_status_start_response)
                return WSGIApp(app)
            raise ForwardRequestException(factory=factory)
    app = TestForwardRequestMiddleware(error_docs_app)
    app = TestApp(RecursiveMiddleware(app))
    res = app.get('')
    assert res.headers['content-type'] == 'text/plain'
    assert res.status == '200 OK'
    assert 'requested page returned' in res
    res = app.get('/error')
    assert res.headers['content-type'] == 'text/plain'
    assert res.status == '200 OK'
    assert 'Page not found' in res
    res = app.get('/not_found', status=404)
    assert res.headers['content-type'] == 'text/plain'
    assert res.status == '404 Not Found'
    assert 'Page not found' in res
    try:
        res = app.get('/recurse')
    except AssertionError as e:
        if str(e).startswith('Forwarding loop detected'):
            pass
        else:
            raise AssertionError('Failed to detect forwarding loop')