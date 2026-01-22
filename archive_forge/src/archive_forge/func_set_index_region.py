import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def set_index_region(self, start, count, data):
    byte_start = self.index_element_size * start
    byte_count = self.index_element_size * count
    ptr_type = ctypes.POINTER(self.index_c_type * count)
    map_ptr = self.index_buffer.map_range(byte_start, byte_count, ptr_type)
    map_ptr[:] = data
    self.index_buffer.unmap()