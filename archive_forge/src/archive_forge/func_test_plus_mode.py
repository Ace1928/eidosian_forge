import os
import shutil
import sys
import tempfile
import unittest
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
def test_plus_mode(self):
    with open(self.tmpfilepath, 'w+') as f:
        f.write(self.teststring)
        f.seek(0)
        new_f = pickle.loads(cloudpickle.dumps(f))
        self.assertEqual(self.teststring, new_f.read())
    os.remove(self.tmpfilepath)