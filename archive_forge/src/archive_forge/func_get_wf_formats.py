import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def get_wf_formats(config):
    default_dpi = {'png': 80, 'hires.png': 200, 'pdf': 200}
    formats = []
    wf_formats = config.wf_formats
    if isinstance(wf_formats, (str, bytes)):
        wf_formats = wf_formats.split(',')
    for fmt in wf_formats:
        if isinstance(fmt, (str, bytes)):
            if ':' in fmt:
                suffix, dpi = fmt.split(':')
                formats.append((str(suffix), int(dpi)))
            else:
                formats.append((fmt, default_dpi.get(fmt, 80)))
        elif isinstance(fmt, (tuple, list)) and len(fmt) == 2:
            formats.append((str(fmt[0]), int(fmt[1])))
        else:
            raise GraphError('invalid image format "%r" in wf_formats' % fmt)
    return formats