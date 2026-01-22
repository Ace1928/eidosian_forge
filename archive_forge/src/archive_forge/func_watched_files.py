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
def watched_files(self, include_globs=True):
    """
        Yield all files that need to be watched, including module files and
        files within globs.
        """
    yield from iter_all_python_module_files()
    yield from self.extra_files
    if include_globs:
        for directory, patterns in self.directory_globs.items():
            for pattern in patterns:
                yield from directory.glob(pattern)