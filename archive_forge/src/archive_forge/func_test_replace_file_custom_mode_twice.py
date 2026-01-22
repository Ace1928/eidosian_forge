import os.path
import stat
from neutron_lib.tests import _base as base
from neutron_lib.utils import file
def test_replace_file_custom_mode_twice(self):
    file_mode = 466
    file.replace_file(self.file_name, self.data, file_mode)
    self.data = 'new data to copy'
    file_mode = 511
    file.replace_file(self.file_name, self.data, file_mode)
    self._verify_result(file_mode)