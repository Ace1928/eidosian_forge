import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def optional_str(x):
    return str(x) if x is not None else None