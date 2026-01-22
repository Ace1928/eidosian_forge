import logging
import shutil
import sys
from pathlib import Path
from . import EXE_FORMAT
from .config import Config
from .python_helper import PythonExecutable
def setup_python(dry_run: bool, force: bool, init: bool):
    """
        Sets up Python venv for deployment, and return a wrapper around the venv environment
    """
    python = None
    response = 'yes'
    if not PythonExecutable.is_venv() and (not force) and (not dry_run) and (not init):
        response = input('You are not using a virtual environment. pyside6-deploy needs to install a few Python packages for deployment to work seamlessly. \nProceed? [Y/n]')
    if response.lower() in ['no', 'n']:
        print('[DEPLOY] Exiting ...')
        sys.exit(0)
    python = PythonExecutable(dry_run=dry_run)
    logging.info(f'[DEPLOY] Using python at {sys.executable}')
    return python