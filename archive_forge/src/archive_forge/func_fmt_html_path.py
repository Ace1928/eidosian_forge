import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def fmt_html_path(path):
    if path is None:
        return path
    else:
        return os.path.abspath(path)