import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest
def testEverythingEphemeralSetattr(self):
    """
        Verify that _EverythingEphemeral.__setattr__ won't affect __main__.
        """
    self.fakeMain.testMainModSetattr = 1
    dirname = self.mktemp()
    os.mkdir(dirname)
    filename = os.path.join(dirname, 'persisttest.ee_setattr')
    with open(filename, 'w') as f:
        f.write('import __main__\n')
        f.write('__main__.testMainModSetattr = 2\n')
        f.write('app = None\n')
    sob.load(filename, 'source')
    self.assertEqual(self.fakeMain.testMainModSetattr, 1)