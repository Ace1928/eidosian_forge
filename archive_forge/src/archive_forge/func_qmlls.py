import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def qmlls():
    qt_tool_wrapper('qmlls', sys.argv[1:])