from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testNonStringDictCompletions(self):
    completions = completion.Completions({10: 'green', 3.14: 'yellow', ('t1', 't2'): 'pink'})
    self.assertIn('10', completions)
    self.assertIn('3.14', completions)
    self.assertIn("('t1', 't2')", completions)
    self.assertNotIn('green', completions)
    self.assertNotIn('yellow', completions)
    self.assertNotIn('pink', completions)