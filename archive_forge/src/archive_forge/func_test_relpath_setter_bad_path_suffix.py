from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_relpath_setter_bad_path_suffix(self):

    def fake_app(environ, start_response):
        self.fail('The app should never be called when the path is wrong')
    wrapped_app = wsgi.RelpathSetter(fake_app, prefix='/abc/', path_var='FOO')
    iterable = wrapped_app({'FOO': '/abc/xyz/.bzr/AAA'}, self.start_response)
    self.read_response(iterable)
    self.assertTrue(self.status.startswith('404'))