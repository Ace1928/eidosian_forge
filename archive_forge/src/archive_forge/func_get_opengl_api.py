import warnings
from ctypes import c_char_p, cast
from pyglet.gl.gl import GL_EXTENSIONS, GL_RENDERER, GL_VENDOR, GL_VERSION
from pyglet.gl.gl import GL_MAJOR_VERSION, GL_MINOR_VERSION, GLint
from pyglet.gl.lib import GLException
from pyglet.util import asstr
def get_opengl_api(self):
    """Determine the OpenGL API version.
        Usually ``gl`` or ``gles``.

        :rtype: str
        """
    if not self._have_context:
        warnings.warn('No GL context created yet.')
    return self.opengl_api