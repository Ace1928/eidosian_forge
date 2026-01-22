import contextlib
import os
import pathlib
import shutil
import stat
import sys
import zipfile
import {module}
def get_interpreter(archive):
    with _maybe_open(archive, 'rb') as f:
        if f.read(2) == b'#!':
            return f.readline().strip().decode(shebang_encoding)