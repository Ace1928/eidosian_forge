from Xlib import X, Xatom, Xutil
from Xlib.protocol import request, rq
from Xlib.xobject import resource, colormap, cursor, fontable, icccm
def put_pil_image(self, gc, x, y, image, onerror=None):
    width, height = image.size
    if image.mode == '1':
        format = X.XYBitmap
        depth = 1
        if self.display.info.bitmap_format_bit_order == 0:
            rawmode = '1;R'
        else:
            rawmode = '1'
        pad = self.display.info.bitmap_format_scanline_pad
        stride = roundup(width, pad) >> 3
    elif image.mode == 'RGB':
        format = X.ZPixmap
        depth = 24
        if self.display.info.image_byte_order == 0:
            rawmode = 'BGRX'
        else:
            rawmode = 'RGBX'
        pad = self.display.info.bitmap_format_scanline_pad
        unit = self.display.info.bitmap_format_scanline_unit
        stride = roundup(width * unit, pad) >> 3
    else:
        raise ValueError('Unknown data format')
    maxlen = (self.display.info.max_request_length << 2) - request.PutImage._request.static_size
    split = maxlen // stride
    x1 = 0
    x2 = width
    y1 = 0
    while y1 < height:
        h = min(height, split)
        if h < height:
            subimage = image.crop((x1, y1, x2, y1 + h))
        else:
            subimage = image
        w, h = subimage.size
        data = subimage.tostring('raw', rawmode, stride, 0)
        self.put_image(gc, x, y, w, h, format, depth, 0, data)
        y1 = y1 + h
        y = y + h