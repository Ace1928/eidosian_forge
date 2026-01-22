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
def testHelpTextMultipleKeywoardArgumentsWithShortArgs(self):
    component = tc.fn_with_multiple_defaults
    t = trace.FireTrace(component, name='shortargs')
    help_screen = helptext.HelpText(component, t)
    self.assertIn(formatting.Bold('NAME') + '\n    shortargs', help_screen)
    self.assertIn(formatting.Bold('SYNOPSIS') + '\n    shortargs <flags>', help_screen)
    self.assertIn(formatting.Bold('FLAGS') + '\n    -f, --first', help_screen)
    self.assertIn('\n    --last', help_screen)
    self.assertIn('\n    --late', help_screen)