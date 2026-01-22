import os
import pytest
from monty.os import cd, makedirs_p
from monty.os.path import find_exts, zpath
def test_makedirs_p(self):
    makedirs_p(self.test_dir_path)
    assert os.path.exists(self.test_dir_path)
    makedirs_p(self.test_dir_path)
    with pytest.raises(OSError):
        makedirs_p(os.path.join(test_dir, 'myfile_txt'))