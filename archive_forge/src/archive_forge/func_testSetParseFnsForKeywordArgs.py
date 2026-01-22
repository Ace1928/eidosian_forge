from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFnsForKeywordArgs(self):
    self.assertEqual(core.Fire(WithKwargs, command=['example6']), ('default', 0))
    self.assertEqual(core.Fire(WithKwargs, command=['example6', '--herring', '"red"']), ('default', 0))
    self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', 'train']), ('train', 0))
    self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', '3']), ('3', 0))
    self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', '-1', '--count', '10']), ('-1', 10))
    self.assertEqual(core.Fire(WithKwargs, command=['example6', '--count', '-2']), ('default', -2))