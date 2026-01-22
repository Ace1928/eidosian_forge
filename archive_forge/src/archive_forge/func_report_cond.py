from nose.plugins.multiprocess import MultiProcessTestRunner  # @UnresolvedImport
from nose.plugins.base import Plugin  # @UnresolvedImport
import sys
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from contextlib import contextmanager
from io import StringIO
import traceback
def report_cond(self, cond, test, captured_output, error=''):
    """
        @param cond: fail, error, ok
        """
    address = self._get_test_address(test)
    error_contents = self.get_io_from_error(error)
    try:
        time_str = '%.2f' % (time.time() - test._pydev_start_time)
    except:
        time_str = '?'
    pydev_runfiles_xml_rpc.notifyTest(cond, captured_output, error_contents, address[0], address[1], time_str)