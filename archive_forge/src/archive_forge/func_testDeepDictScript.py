from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testDeepDictScript(self):
    deepdict = {'level1': {'level2': {'level3': {'level4': {}}}}}
    script = completion.Script('deepdict', deepdict)
    self.assertIn('level1', script)
    self.assertIn('level2', script)
    self.assertIn('level3', script)
    self.assertNotIn('level4', script)