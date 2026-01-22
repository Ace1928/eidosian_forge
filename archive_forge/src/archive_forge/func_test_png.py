import base64
import sys
import zlib
import pytest
from . import constants, pixbuf
def test_png():
    pixbuf_obj, format_name = pixbuf.decode_to_pixbuf(JPEG_BYTES)
    assert format_name == 'jpeg'
    assert_decoded(pixbuf.pixbuf_to_cairo_slices(pixbuf_obj), constants.FORMAT_RGB24, b'\xff\x00\x80\xff')