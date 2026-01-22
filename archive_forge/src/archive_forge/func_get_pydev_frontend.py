from __future__ import print_function
import os
import sys
import codeop
import traceback
from IPython.core.error import UsageError
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.usage import default_banner_parts
from IPython.utils.strdispatch import StrDispatch
import IPython.core.release as IPythonRelease
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core import release
from _pydev_bundle.pydev_imports import xmlrpclib
def get_pydev_frontend(pydev_host, pydev_client_port):
    if _PyDevFrontEndContainer._instance is None:
        _PyDevFrontEndContainer._instance = _PyDevFrontEnd()
    if _PyDevFrontEndContainer._last_host_port != (pydev_host, pydev_client_port):
        _PyDevFrontEndContainer._last_host_port = (pydev_host, pydev_client_port)
        _PyDevFrontEndContainer._instance.ipython.hooks['editor'] = create_editor_hook(pydev_host, pydev_client_port)
    return _PyDevFrontEndContainer._instance