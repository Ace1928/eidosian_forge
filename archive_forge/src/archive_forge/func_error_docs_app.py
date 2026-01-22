from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
def error_docs_app(environ, start_response):
    if environ['PATH_INFO'] == '/not_found':
        start_response('404 Not found', [('Content-type', 'text/plain')])
        return [b'Not found']
    elif environ['PATH_INFO'] == '/error':
        start_response('200 OK', [('Content-type', 'text/plain')])
        return [b'Page not found']
    elif environ['PATH_INFO'] == '/recurse':
        raise ForwardRequestException('/recurse')
    else:
        return simple_app(environ, start_response)