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
def autosave(self, arg_s):
    """Set the autosave interval in the notebook (in seconds).

        The default value is 120, or two minutes.
        ``%autosave 0`` will disable autosave.

        This magic only has an effect when called from the notebook interface.
        It has no effect when called in a startup file.
        """
    try:
        interval = int(arg_s)
    except ValueError as e:
        raise UsageError('%%autosave requires an integer, got %r' % arg_s) from e
    milliseconds = 1000 * interval
    display(Javascript('IPython.notebook.set_autosave_interval(%i)' % milliseconds), include=['application/javascript'])
    if interval:
        print('Autosaving every %i seconds' % interval)
    else:
        print('Autosave disabled')