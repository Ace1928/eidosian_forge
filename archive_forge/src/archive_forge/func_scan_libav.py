from .utils import check_output, where
import os
import warnings
import numpy as np
def scan_libav():
    global _LIBAV_MAJOR_VERSION
    global _LIBAV_MINOR_VERSION
    _LIBAV_MAJOR_VERSION = '0'
    _LIBAV_MINOR_VERSION = '0'
    try:
        version = check_output([os.path.join(_AVCONV_PATH, _AVCONV_APPLICATION), '-version'])
        firstline = version.split(b'\n')[0]
        firstlineparts = firstline.split(b' ')
        version = ''
        if firstlineparts[1].strip() == b'version':
            version = firstlineparts[2].split('.')[0]
        else:
            version = firstlineparts[1].split(b'-')[0]
        version = version.split(b'_')[0]
        versionparts = version.split(b'.')
        if versionparts[0].decode()[0] == 'v':
            _LIBAV_MAJOR_VERSION = versionparts[0].decode()[1:]
        else:
            _LIBAV_MAJOR_VERSION = str(versionparts[0].decode())
            _LIBAV_MINOR_VERSION = str(versionparts[1].decode())
    except:
        pass