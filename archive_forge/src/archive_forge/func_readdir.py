import errno
import locale
import logging
import os
import stat
import sys
import time
from optparse import Option, OptionParser
import nibabel as nib
import nibabel.dft as dft
def readdir(self, path, fh):
    logger.info(f'readdir {path}')
    matched_path = self.match_path(path)
    if matched_path is None:
        return -errno.ENOENT
    logger.debug(f'matched {matched_path}')
    fnames = [k.encode('ascii', 'replace') for k in matched_path.keys()]
    fnames.extend(('.', '..'))
    return [fuse.Direntry(f) for f in fnames]