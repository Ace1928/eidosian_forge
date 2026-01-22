import io
import math
import os
import typing
import weakref
def find_buffer_by_name(name):
    for buffer, (name_set, _, _) in font_buffers.items():
        if name in name_set:
            return buffer
    return None