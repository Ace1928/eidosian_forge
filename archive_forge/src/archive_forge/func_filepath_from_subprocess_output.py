import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def filepath_from_subprocess_output(output):
    """
    Convert `bytes` in the encoding used by a subprocess into a filesystem-appropriate `str`.

    Inherited from `exec_command`, and possibly incorrect.
    """
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    output = output.decode(mylocale, errors='replace')
    output = output.replace('\r\n', '\n')
    if output[-1:] == '\n':
        output = output[:-1]
    return output