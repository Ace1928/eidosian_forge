from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testFnFishScript(self):
    script = completion.Script('identity', tc.identity, shell='fish')
    self.assertIn('arg1', script)
    self.assertIn('arg2', script)
    self.assertIn('arg3', script)
    self.assertIn('arg4', script)