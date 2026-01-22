import shutil
import tempfile
import unittest
def tempfilename(self):
    with tempfile.NamedTemporaryFile(dir=self.tmpdir) as nf:
        return nf.name