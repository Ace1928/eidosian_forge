import os
import binascii
from io import BytesIO
from reportlab import rl_config
from reportlab.lib.utils import ImageReader, isUnicode
from reportlab.lib.rl_accel import asciiBase85Encode, asciiBase85Decode
def cacheImageFile(filename, returnInMemory=0, IMG=None):
    """Processes image as if for encoding, saves to a file with .a85 extension."""
    cachedname = os.path.splitext(filename)[0] + (rl_config.useA85 and '.a85' or '.bin')
    if filename == cachedname:
        if cachedImageExists(filename):
            from reportlab.lib.utils import open_for_read
            if returnInMemory:
                return filter(None, open_for_read(cachedname).read().split('\r\n'))
        else:
            raise IOError('No such cached image %s' % filename)
    else:
        if rl_config.useA85:
            code = makeA85Image(filename, IMG)
        else:
            code = makeRawImage(filename, IMG)
        if returnInMemory:
            return code
        f = open(cachedname, 'wb')
        f.write('\r\n'.join(code) + '\r\n')
        f.close()
        if rl_config.verbose:
            print('cached image as %s' % cachedname)