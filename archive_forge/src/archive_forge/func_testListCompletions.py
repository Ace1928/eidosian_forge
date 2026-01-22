from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testListCompletions(self):
    completions = completion.Completions(['red', 'green', 'blue'])
    self.assertIn('0', completions)
    self.assertIn('1', completions)
    self.assertIn('2', completions)
    self.assertNotIn('3', completions)