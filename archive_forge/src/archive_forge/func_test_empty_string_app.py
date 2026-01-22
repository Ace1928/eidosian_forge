import sys
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def test_empty_string_app(environ, start_response):
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    return [b'Hello', b'', b' ', b'', b'world']