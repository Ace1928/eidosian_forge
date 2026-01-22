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
@testutils.skipIf(six.PY2, 'Python 2 does not support required name-only arguments.')
def testUsageOutputFunctionMixedDefaults(self):
    component = tc.py3.HelpTextComponent().identity
    t = trace.FireTrace(component, name='FunctionMixedDefaults')
    usage_output = helptext.UsageText(component, trace=t, verbose=False)
    expected_output = '\n    Usage: FunctionMixedDefaults <flags>\n      optional flags:        --beta\n      required flags:        --alpha\n\n    For detailed information on this command, run:\n      FunctionMixedDefaults --help'
    expected_output = textwrap.dedent(expected_output).lstrip('\n')
    self.assertEqual(expected_output, usage_output)