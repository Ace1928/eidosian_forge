import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def init_virtual_env():
    """PYSIDE-2251: Enable running from a non-activated virtual environment
       as is the case for Visual Studio Code by setting the VIRTUAL_ENV
       variable which is used by the Qt Designer plugin."""
    if is_virtual_env() and (not os.environ.get(VIRTUAL_ENV)):
        os.environ[VIRTUAL_ENV] = sys.prefix