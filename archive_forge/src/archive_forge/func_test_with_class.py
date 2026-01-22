import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_with_class(self):

    @deprecated(version='test')
    class foo(object):

        def __init__(self):
            logger.warning('yeah')
    self.assertIs(type(foo), type)
    self.assertRegex(foo.__doc__, '.. deprecated:: test\\n   This class \\(.*\\.foo\\) has been deprecated')
    DEP_OUT = StringIO()
    FCN_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo'):
        with LoggingIntercept(FCN_OUT, 'local'):
            foo()
    self.assertIn('yeah', FCN_OUT.getvalue())
    self.assertNotIn('DEPRECATED', FCN_OUT.getvalue())
    self.assertRegex(DEP_OUT.getvalue().replace('\n', ' '), 'DEPRECATED: This class \\(.*\\.foo\\) has been deprecated.*\\(deprecated in test\\)')