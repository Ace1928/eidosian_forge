import os
import subprocess
from IPython.core.error import TryNext
import IPython.utils.py3compat as py3compat
def osx_clipboard_get() -> str:
    """ Get the clipboard's text on OS X.
    """
    p = subprocess.Popen(['pbpaste', '-Prefer', 'ascii'], stdout=subprocess.PIPE)
    bytes_, stderr = p.communicate()
    bytes_ = bytes_.replace(b'\r', b'\n')
    text = py3compat.decode(bytes_)
    return text