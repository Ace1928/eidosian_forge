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
def get_opt_parser():
    p = OptionParser(usage='{} [OPTIONS] <DIRECTORY CONTAINING DICOMSs> <mount point>'.format(os.path.basename(sys.argv[0])), version='%prog ' + nib.__version__)
    p.add_options([Option('-v', '--verbose', action='count', dest='verbose', default=0, help='make noise.  Could be specified multiple times')])
    p.add_options([Option('-L', '--follow-links', action='store_true', dest='followlinks', default=False, help='Follow symbolic links in DICOM directory')])
    return p