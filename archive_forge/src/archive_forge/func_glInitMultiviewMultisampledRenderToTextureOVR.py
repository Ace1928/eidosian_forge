from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.OVR.multiview_multisampled_render_to_texture import *
from OpenGL.raw.GLES2.OVR.multiview_multisampled_render_to_texture import _EXTENSION_NAME
def glInitMultiviewMultisampledRenderToTextureOVR():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)