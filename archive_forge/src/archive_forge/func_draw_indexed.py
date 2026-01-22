import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def draw_indexed(size, mode, indices, **data):
    """Draw a primitive with indexed vertices immediately.

    :warning: This function is deprecated as of 2.0.4, and will be removed
              in the next release.

    :Parameters:
        `size` : int
            Number of vertices given
        `mode` : int
            OpenGL drawing mode, e.g. ``GL_TRIANGLES``
        `indices` : sequence of int
            Sequence of integers giving indices into the vertex list.
        `**data` : keyword arguments for passing vertex attribute data.
            The keyword should be the vertex attribute name, and the
            argument should be a tuple of (format, data). For example:
            `position=('f', array)`

    """
    vao_id = GLuint()
    glGenVertexArrays(1, vao_id)
    glBindVertexArray(vao_id)
    program = get_default_shader()
    program.use()
    buffers = []
    for name, (fmt, array) in data.items():
        location = program.attributes[name]['location']
        count = program.attributes[name]['count']
        gl_type = vertexdomain._gl_types[fmt[0]]
        normalize = 'n' in fmt
        attribute = shader.Attribute(name, location, count, gl_type, normalize)
        assert size == len(array) // attribute.count, 'Data for %s is incorrect length' % fmt
        buffer = BufferObject(size * attribute.stride)
        data = (attribute.c_type * len(array))(*array)
        buffer.set_data(data)
        attribute.enable()
        attribute.set_pointer(buffer.ptr)
        buffers.append(buffer)
    if size <= 255:
        index_type = GL_UNSIGNED_BYTE
        index_c_type = ctypes.c_ubyte
    elif size <= 65535:
        index_type = GL_UNSIGNED_SHORT
        index_c_type = ctypes.c_ushort
    else:
        index_type = GL_UNSIGNED_INT
        index_c_type = ctypes.c_uint
    index_array = (index_c_type * len(indices))(*indices)
    index_buffer = BufferObject(ctypes.sizeof(index_array))
    index_buffer.set_data(index_array)
    index_buffer.bind_to_index_buffer()
    glDrawElements(mode, len(indices), index_type, 0)
    glFlush()
    program.stop()
    del buffers
    del index_buffer
    glBindVertexArray(0)
    glDeleteVertexArrays(1, vao_id)