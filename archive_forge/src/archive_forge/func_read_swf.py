import os
import zlib
import time  # noqa
import logging
import numpy as np
def read_swf(filename):
    """Read all images from an SWF (shockwave flash) file. Returns a list
    of numpy arrays.

    Limitation: only read the PNG encoded images (not the JPG encoded ones).
    """
    if not os.path.isfile(filename):
        raise IOError('File not found: ' + str(filename))
    images = []
    fp = open(filename, 'rb')
    bb = fp.read()
    try:
        tmp = bb[0:3].decode('ascii', 'ignore')
        if tmp.upper() == 'FWS':
            pass
        elif tmp.upper() == 'CWS':
            bb = bb[:8] + zlib.decompress(bb[8:])
        else:
            raise IOError('Not a valid SWF file: ' + str(filename))
        i = 8
        nbits = bits2int(bb[i:i + 1], 5)
        nbits = 5 + nbits * 4
        Lrect = nbits / 8.0
        if Lrect % 1:
            Lrect += 1
        Lrect = int(Lrect)
        i += Lrect + 4
        counter = 0
        while True:
            counter += 1
            head = bb[i:i + 6]
            if not head:
                break
            T, L1, L2 = get_type_and_len(head)
            if not L2:
                logger.warning('Invalid tag length, could not proceed')
                break
            if T in [20, 36]:
                im = read_pixels(bb, i + 6, T, L1)
                if im is not None:
                    images.append(im)
            elif T in [6, 21, 35, 90]:
                logger.warning('Ignoring JPEG image: cannot read JPEG.')
            else:
                pass
            if T == 0:
                break
            i += L2
    finally:
        fp.close()
    return images