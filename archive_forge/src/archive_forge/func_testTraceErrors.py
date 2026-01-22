from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import fire
from fire import test_components as tc
from fire import testutils
import mock
import six
def testTraceErrors(self):
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['a1'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['--arg1=a1'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['a1', 'a2', '-', 'run', 'b1'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2', '-', 'run b1'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['a1', 'a2', '-', 'run', 'b1', 'b2', 'b3'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2', '-', 'run', 'b1', 'b2', 'b3'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['a1', 'a2', '-', 'jog'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2', '-', 'jog'])