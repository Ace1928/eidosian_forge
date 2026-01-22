import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
def getGraphicsControlExt(self, duration=0.1, dispose=2):
    """Graphics Control Extension. A sort of header at the start of
        each image. Specifies duration and transparancy.

        Dispose
        -------
          * 0 - No disposal specified.
          * 1 - Do not dispose. The graphic is to be left in place.
          * 2 - Restore to background color. The area used by the graphic
            must be restored to the background color.
          * 3 - Restore to previous. The decoder is required to restore the
            area overwritten by the graphic with what was there prior to
            rendering the graphic.
          * 4-7 -To be defined.
        """
    bb = b'!\xf9\x04'
    bb += chr((dispose & 3) << 2).encode('utf-8')
    bb += intToBin(int(duration * 100 + 0.5))
    bb += b'\x00'
    bb += b'\x00'
    return bb