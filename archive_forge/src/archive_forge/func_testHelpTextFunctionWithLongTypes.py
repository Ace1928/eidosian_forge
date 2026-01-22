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
@testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
def testHelpTextFunctionWithLongTypes(self):
    component = tc.py3.WithTypes().long_type
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='long_type'))
    self.assertIn('NAME\n    long_type', help_screen)
    self.assertIn('SYNOPSIS\n    long_type LONG_OBJ', help_screen)
    self.assertNotIn('DESCRIPTION', help_screen)
    self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)