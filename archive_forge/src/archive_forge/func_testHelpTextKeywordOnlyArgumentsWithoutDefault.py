from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import textwrap
from fire import formatting
from fire import helptext
from fire import test_components as tc
from fire import testutils
from fire import trace
import six
@testutils.skipIf(six.PY2, 'Python 2 does not support keyword-only arguments.')
def testHelpTextKeywordOnlyArgumentsWithoutDefault(self):
    component = tc.py3.KeywordOnly.double
    output = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'double'))
    self.assertIn('NAME\n    double', output)
    self.assertIn('FLAGS\n    -c, --count=COUNT (required)', output)