import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
def handler_by_plugin_name(self, name):
    plugin = plugins.FormatHandler.by_name(name)
    if plugin:
        try:
            return plugin.load()
        except ImportError:
            return None
    else:
        raise RuntimeError('No handler of name %s found' % (name,))