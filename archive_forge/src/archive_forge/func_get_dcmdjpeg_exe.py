import os
import sys
import logging
import subprocess
from ..core import Format, BaseProgressIndicator, StdoutProgressIndicator
from ..core import read_n_bytes
def get_dcmdjpeg_exe():
    fname = 'dcmdjpeg' + '.exe' * sys.platform.startswith('win')
    for dir in ('c:\\dcmtk', 'c:\\Program Files', 'c:\\Program Files\\dcmtk', 'c:\\Program Files (x86)\\dcmtk'):
        filename = os.path.join(dir, fname)
        if os.path.isfile(filename):
            return [filename]
    try:
        subprocess.check_call([fname, '--version'])
        return [fname]
    except Exception:
        return None