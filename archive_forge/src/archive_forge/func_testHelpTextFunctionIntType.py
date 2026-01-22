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
def testHelpTextFunctionIntType(self):
    component = int
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'int'))
    self.assertIn('NAME\n    int', help_screen)
    self.assertIn('SYNOPSIS\n    int', help_screen)
    self.assertIn('DESCRIPTION\n', help_screen)