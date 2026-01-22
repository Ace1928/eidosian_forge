from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_smart_wsgi_app_request_and_response(self):
    transport = memory.MemoryTransport()
    transport.put_bytes('foo', b'some bytes')
    wsgi_app = wsgi.SmartWSGIApp(transport)
    wsgi_app.make_request = self._fake_make_request
    fake_input = BytesIO(b'fake request')
    environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo'})
    iterable = wsgi_app(environ, self.start_response)
    response = self.read_response(iterable)
    self.assertEqual('200 OK', self.status)
    self.assertEqual(b'got bytes: fake request', response)