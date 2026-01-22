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
def testHelpTextFunctionMixedDefaults(self):
    component = tc.py3.HelpTextComponent().identity
    t = trace.FireTrace(component, name='FunctionMixedDefaults')
    output = helptext.HelpText(component, trace=t)
    self.assertIn('NAME\n    FunctionMixedDefaults', output)
    self.assertIn('FunctionMixedDefaults <flags>', output)
    self.assertIn('--alpha=ALPHA (required)', output)
    self.assertIn("--beta=BETA\n        Default: '0'", output)