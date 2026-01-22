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
def testHelpTextNoDefaults(self):
    component = tc.NoDefaults
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='NoDefaults'))
    self.assertIn('NAME\n    NoDefaults', help_screen)
    self.assertIn('SYNOPSIS\n    NoDefaults', help_screen)
    self.assertNotIn('DESCRIPTION', help_screen)
    self.assertNotIn('NOTES', help_screen)