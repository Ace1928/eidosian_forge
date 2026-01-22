import ctypes
from pyglet.gl import *
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import BufferObject, AttributeBufferObject
def set_attribute_data(self, name, data):
    attribute = self.domain.attribute_names[name]
    array_start = attribute.count * self.start
    array_end = attribute.count * self.count + array_start
    try:
        attribute.buffer.data[array_start:array_end] = data
        attribute.buffer.invalidate_region(self.start, self.count)
    except ValueError:
        raise ValueError(f"Invalid data size for '{name}'. Expected {array_end - array_start}, got {len(data)}.") from None