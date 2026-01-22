import os
import shutil
import unittest
from monty.tempfile import ScratchDir
def test_no_copy(self):
    with ScratchDir(self.scratch_root, copy_from_current_on_enter=False, copy_to_current_on_exit=False) as d:
        with open('scratch_text', 'w') as f:
            f.write('write')
        files = os.listdir(d)
        assert 'scratch_text' in files
        assert 'empty_file.txt' not in files
    assert not os.path.exists(d)
    files = os.listdir('.')
    assert 'scratch_text' not in files