import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType
from zipimport import zipimporter
import django
from django.apps import apps
from django.core.signals import request_finished
from django.dispatch import Signal
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
def get_child_arguments():
    """
    Return the executable. This contains a workaround for Windows if the
    executable is reported to not have the .exe extension which can cause bugs
    on reloading.
    """
    import __main__
    py_script = Path(sys.argv[0])
    exe_entrypoint = py_script.with_suffix('.exe')
    args = [sys.executable] + ['-W%s' % o for o in sys.warnoptions]
    if sys.implementation.name == 'cpython':
        args.extend((f'-X{key}' if value is True else f'-X{key}={value}' for key, value in sys._xoptions.items()))
    if getattr(__main__, '__spec__', None) is not None and (not exe_entrypoint.exists()):
        spec = __main__.__spec__
        if (spec.name == '__main__' or spec.name.endswith('.__main__')) and spec.parent:
            name = spec.parent
        else:
            name = spec.name
        args += ['-m', name]
        args += sys.argv[1:]
    elif not py_script.exists():
        if exe_entrypoint.exists():
            return [exe_entrypoint, *sys.argv[1:]]
        script_entrypoint = py_script.with_name('%s-script.py' % py_script.name)
        if script_entrypoint.exists():
            return [*args, script_entrypoint, *sys.argv[1:]]
        raise RuntimeError('Script %s does not exist.' % py_script)
    else:
        args += sys.argv
    return args