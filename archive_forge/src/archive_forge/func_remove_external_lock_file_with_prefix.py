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
def remove_external_lock_file_with_prefix(lock_file_prefix):
    """Partial object generator for the remove lock file function.

    Redefine remove_external_lock_file_with_prefix in each project like so::

        (in nova/utils.py)
        from oslo_concurrency import lockutils

        _prefix = 'nova'
        synchronized = lockutils.synchronized_with_prefix(_prefix)
        lock = lockutils.lock_with_prefix(_prefix)
        lock_cleanup = lockutils.remove_external_lock_file_with_prefix(_prefix)

        (in nova/foo.py)
        from nova import utils

        @utils.synchronized('mylock')
        def bar(self, *args):
            ...

        def baz(self, *args):
            ...
            with utils.lock('mylock'):
                ...
            ...

        <eventually call lock_cleanup('mylock') to clean up>

    The lock_file_prefix argument is used to provide lock files on disk with a
    meaningful prefix.
    """
    return functools.partial(remove_external_lock_file, lock_file_prefix=lock_file_prefix)