import base64
import sys
import zlib
import pytest
from . import constants, pixbuf
def test_gdk():
    if pixbuf.gdk is None:
        pytest.xfail()
    pixbuf_obj, format_name = pixbuf.decode_to_pixbuf(PNG_BYTES)
    assert format_name == 'png'
    assert_decoded(pixbuf.pixbuf_to_cairo_gdk(pixbuf_obj))