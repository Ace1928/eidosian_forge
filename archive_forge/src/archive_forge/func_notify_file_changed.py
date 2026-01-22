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
def notify_file_changed(self, path):
    results = file_changed.send(sender=self, file_path=path)
    logger.debug('%s notified as changed. Signal results: %s.', path, results)
    if not any((res[1] for res in results)):
        trigger_reload(path)