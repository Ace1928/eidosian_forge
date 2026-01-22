import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_write_to_user_data_directory(self):
    """
        write to user data directory

        """
    self.ETSConfig.company = 'Blah'
    dirname = self.ETSConfig.user_data
    path = os.path.join(dirname, 'dummy.txt')
    data = str(time.time())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)
    self.assertEqual(os.path.exists(path), True)
    with open(path, 'r', encoding='utf-8') as f:
        result = f.read()
    os.remove(path)
    self.assertEqual(data, result)