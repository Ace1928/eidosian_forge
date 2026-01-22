import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def safe_index_realloc(self, start, count, new_count):
    """Reallocate indices, resizing the buffers if necessary."""
    try:
        return self.index_allocator.realloc(start, count, new_count)
    except allocation.AllocatorMemoryException as e:
        capacity = _nearest_pow2(e.requested_capacity)
        self.index_buffer.resize(capacity * self.index_element_size)
        self.index_allocator.set_capacity(capacity)
        return self.index_allocator.realloc(start, count, new_count)