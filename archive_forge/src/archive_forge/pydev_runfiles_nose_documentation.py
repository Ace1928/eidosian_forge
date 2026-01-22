from nose.plugins.multiprocess import MultiProcessTestRunner  # @UnresolvedImport
from nose.plugins.base import Plugin  # @UnresolvedImport
import sys
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from contextlib import contextmanager
from io import StringIO
import traceback

        @param cond: fail, error, ok
        