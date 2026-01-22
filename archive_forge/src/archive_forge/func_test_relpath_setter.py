from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_relpath_setter(self):
    calls = []

    def fake_app(environ, start_response):
        calls.append(environ['breezy.relpath'])
    wrapped_app = wsgi.RelpathSetter(fake_app, prefix='/abc/', path_var='FOO')
    wrapped_app({'FOO': '/abc/xyz/.bzr/smart'}, None)
    self.assertEqual(['xyz'], calls)