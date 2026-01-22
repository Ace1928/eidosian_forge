from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testAddCalledCallable(self):
    t = trace.FireTrace('initial object')
    args = ('example', 'args')
    t.AddCalledComponent('result', 'cell', args, 'sample.py', 10, False, action=trace.CALLED_CALLABLE)
    self.assertEqual(str(t), '1. Initial component\n2. Called callable "cell" (sample.py:10)')