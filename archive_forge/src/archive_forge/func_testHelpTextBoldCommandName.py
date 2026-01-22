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
def testHelpTextBoldCommandName(self):
    component = tc.ClassWithDocstring()
    t = trace.FireTrace(component, name='ClassWithDocstring')
    help_screen = helptext.HelpText(component, t)
    self.assertIn(formatting.Bold('NAME') + '\n    ClassWithDocstring', help_screen)
    self.assertIn(formatting.Bold('COMMANDS') + '\n', help_screen)
    self.assertIn(formatting.BoldUnderline('COMMAND') + ' is one of the following:\n', help_screen)
    self.assertIn(formatting.Bold('print_msg') + '\n', help_screen)