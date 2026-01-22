import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def test_write_newer_file(self):
    file_path_1 = self._test_path('abcfile1.txt')
    file_path_2 = self._test_path('abcfile2.txt')
    write_file(file_path_1, 'abc')
    assert os.path.isfile(file_path_1)
    write_newer_file(file_path_2, file_path_1, 'xyz')
    assert os.path.isfile(file_path_2)
    assert os.path.getmtime(file_path_2) > os.path.getmtime(file_path_1)