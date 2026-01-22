import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def get_executable_extension():
    if is_windows():
        return '.bat' if is_anaconda() else '.exe'
    else:
        return ''