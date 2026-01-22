import struct
from pyglet.image.codecs import ImageDecodeException
def read_table_based_image(file, stream, graphics_scope):
    gif_image = GIFImage()
    stream.images.append(gif_image)
    gif_image.delay = graphics_scope.delay
    image_left_position, image_top_position, image_width, image_height, fields = unpack('HHHHB', file)
    local_color_table_flag = fields & 128
    local_color_table_size = fields & 7
    if local_color_table_flag:
        local_color_table = file.read(6 << local_color_table_size)
    lzw_code_size = file.read(1)
    skip_data_sub_blocks(file)