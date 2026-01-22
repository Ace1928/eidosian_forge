import logging
import shutil
import sys
from pathlib import Path
from . import EXE_FORMAT
from .config import Config
from .python_helper import PythonExecutable
def config_option_exists():
    for argument in sys.argv:
        if any((item in argument for item in ['--config-file', '-c'])):
            return True
    return False