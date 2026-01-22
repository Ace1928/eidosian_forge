import os
import tempfile
from fire import __main__
from fire import testutils
def testFileNameModuleDuplication(self):
    with self.assertOutputMatches('gettempdir'):
        dirname = os.path.dirname(self.file.name)
        with testutils.ChangeDirectory(dirname):
            with open('tempfile', 'w'):
                __main__.main(['__main__.py', 'tempfile'])
            os.remove('tempfile')