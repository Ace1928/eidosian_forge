from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testGetCommandWithFlagQuotes(self):
    t = trace.FireTrace('initial object')
    args = ('--example=spaced arg',)
    t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
    self.assertEqual(t.GetCommand(), "--example='spaced arg'")