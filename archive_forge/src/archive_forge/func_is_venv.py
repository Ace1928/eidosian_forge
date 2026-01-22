import ast
import logging
import os
import re
import sys
import warnings
from typing import List
from importlib import util
from importlib.metadata import version
from pathlib import Path
from . import Nuitka, run_command
@staticmethod
def is_venv():
    venv = os.environ.get('VIRTUAL_ENV')
    return True if venv else False