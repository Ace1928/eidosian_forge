import struct
from pyglet.image.codecs import ImageDecodeException
def read_graphic_control_extension(file, stream, graphics_scope):
    block_size, fields, delay_time, transparent_color_index, terminator = unpack('BBHBB', file)
    if block_size != 4:
        raise ImageDecodeException('Incorrect block size')
    if delay_time:
        if delay_time <= 1:
            delay_time = 10
        graphics_scope.delay = float(delay_time) / 100