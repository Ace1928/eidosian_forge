import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def linguist():
    qt_tool_wrapper(ui_tool_binary('linguist'), sys.argv[1:])