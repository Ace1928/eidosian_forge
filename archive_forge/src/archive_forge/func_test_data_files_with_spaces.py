from __future__ import print_function
import os
import fixtures
from pbr.hooks import files
from pbr.tests import base
def test_data_files_with_spaces(self):
    config = dict(files=dict(data_files="\n  'i like spaces' = 'dir with space'/*"))
    files.FilesConfig(config, 'fake_package').run()
    self.assertIn("\n'i like spaces/' = \n 'dir with space/file with spc'", config['files']['data_files'])