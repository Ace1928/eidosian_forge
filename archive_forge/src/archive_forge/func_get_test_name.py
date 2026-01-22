import unittest as python_unittest
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydevd_bundle import pydevd_io
import traceback
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
from io import StringIO
def get_test_name(self, test):
    try:
        try:
            test_name = test.__class__.__name__ + '.' + test._testMethodName
        except AttributeError:
            try:
                test_name = test.__class__.__name__ + '.' + test._TestCase__testMethodName
            except:
                test_name = test.description.split()[1][1:-1] + ' <' + test.description.split()[0] + '>'
    except:
        traceback.print_exc()
        return '<unable to get test name>'
    return test_name