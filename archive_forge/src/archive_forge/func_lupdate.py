import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def lupdate():
    qt_tool_wrapper('lupdate', sys.argv[1:])