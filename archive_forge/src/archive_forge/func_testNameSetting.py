import os
import tempfile
from fire import __main__
from fire import testutils
def testNameSetting(self):
    with self.assertOutputMatches('gettempdir'):
        __main__.main(['__main__.py', 'tempfile'])