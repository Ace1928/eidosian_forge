from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testClassFishScript(self):
    script = completion.Script('', tc.MixedDefaults, shell='fish')
    self.assertIn('ten', script)
    self.assertIn('sum', script)
    self.assertIn('identity', script)
    self.assertIn('alpha', script)
    self.assertIn('beta', script)