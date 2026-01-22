import contextlib
import errno
import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import fasteners
from oslo_config import cfg
from oslo_utils import reflection
from oslo_utils import timeutils
from oslo_concurrency._i18n import _
def external_lock(name, lock_file_prefix=None, lock_path=None):
    lock_file_path = _get_lock_path(name, lock_file_prefix, lock_path)
    return InterProcessLock(lock_file_path)