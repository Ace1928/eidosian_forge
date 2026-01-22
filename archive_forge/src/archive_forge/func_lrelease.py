import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def lrelease():
    qt_tool_wrapper('lrelease', sys.argv[1:])