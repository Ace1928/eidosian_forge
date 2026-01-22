import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def find_editor() -> Optional[str]:
    """
    Used to set cmd2.Cmd.DEFAULT_EDITOR. If EDITOR env variable is set, that will be used.
    Otherwise the function will look for a known editor in directories specified by PATH env variable.
    :return: Default editor or None
    """
    editor = os.environ.get('EDITOR')
    if not editor:
        if sys.platform[:3] == 'win':
            editors = ['code.cmd', 'notepad++.exe', 'notepad.exe']
        else:
            editors = ['vim', 'vi', 'emacs', 'nano', 'pico', 'joe', 'code', 'subl', 'atom', 'gedit', 'geany', 'kate']
        env_path = os.getenv('PATH')
        if env_path is None:
            paths = []
        else:
            paths = [p for p in env_path.split(os.path.pathsep) if not os.path.islink(p)]
        for editor, path in itertools.product(editors, paths):
            editor_path = os.path.join(path, editor)
            if os.path.isfile(editor_path) and os.access(editor_path, os.X_OK):
                if sys.platform[:3] == 'win':
                    editor = os.path.splitext(editor)[0]
                break
        else:
            editor = None
    return editor