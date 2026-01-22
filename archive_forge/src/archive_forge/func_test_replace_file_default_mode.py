import os.path
import stat
from neutron_lib.tests import _base as base
from neutron_lib.utils import file
def test_replace_file_default_mode(self):
    file_mode = 420
    file.replace_file(self.file_name, self.data)
    self._verify_result(file_mode)