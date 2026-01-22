from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_smart_wsgi_app_uses_given_relpath(self):
    transport = FakeTransport()
    wsgi_app = wsgi.SmartWSGIApp(transport)
    wsgi_app.backing_transport = transport
    wsgi_app.make_request = self._fake_make_request
    fake_input = BytesIO(b'fake request')
    environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo/bar'})
    iterable = wsgi_app(environ, self.start_response)
    response = self.read_response(iterable)
    self.assertEqual([('clone', 'foo/bar/')], transport.calls)