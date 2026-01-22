from wsgiref import simple_server
from oslo_serialization import jsonutils
from keystonemiddleware import auth_token
def echo_app(environ, start_response):
    """A WSGI application that echoes the CGI environment back to the user."""
    start_response('200 OK', [('Content-Type', 'application/json')])
    environment = dict(((k, v) for k, v in environ.items() if k.startswith('HTTP_X_')))
    yield jsonutils.dumps(environment)