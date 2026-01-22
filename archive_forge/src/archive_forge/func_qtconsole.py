import os
import sys
import warnings
from pathlib import Path
from threading import local
from IPython.core import page, payloadpage
from IPython.core.autocall import ZMQExitAutocall
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magics import CodeMagics, MacroToEdit  # type:ignore[attr-defined]
from IPython.core.usage import default_banner
from IPython.display import Javascript, display
from IPython.utils import openpy
from IPython.utils.process import arg_split, system  # type:ignore[attr-defined]
from jupyter_client.session import Session, extract_header
from jupyter_core.paths import jupyter_runtime_dir
from traitlets import Any, CBool, CBytes, Dict, Instance, Type, default, observe
from ipykernel import connect_qtconsole, get_connection_file, get_connection_info
from ipykernel.displayhook import ZMQShellDisplayHook
from ipykernel.jsonutil import encode_images, json_clean
@line_magic
def qtconsole(self, arg_s):
    """Open a qtconsole connected to this kernel.

        Useful for connecting a qtconsole to running notebooks, for better
        debugging.
        """
    if 'ipyparallel' in sys.modules:
        from ipyparallel import bind_kernel
        bind_kernel()
    try:
        connect_qtconsole(argv=arg_split(arg_s, os.name == 'posix'))
    except Exception as e:
        warnings.warn('Could not start qtconsole: %r' % e, stacklevel=2)
        return