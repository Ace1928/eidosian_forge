from nose.plugins.multiprocess import MultiProcessTestRunner  # @UnresolvedImport
from nose.plugins.base import Plugin  # @UnresolvedImport
import sys
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from contextlib import contextmanager
from io import StringIO
import traceback
def get_io_from_error(self, err):
    if type(err) == type(()):
        if len(err) != 3:
            if len(err) == 2:
                return err[1]
        s = StringIO()
        etype, value, tb = err
        if isinstance(value, str):
            return value
        traceback.print_exception(etype, value, tb, file=s)
        return s.getvalue()
    return err