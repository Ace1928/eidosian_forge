from __future__ import print_function
import os
import fixtures
from pbr.hooks import files
from pbr.tests import base
def test_data_files_with_spaces_subdirectories(self):
    data_files = "\n 'one space/two space' = 'multi space/more spaces'/*"
    expected = "\n'one space/two space/' = \n 'multi space/more spaces/file with spc'"
    config = dict(files=dict(data_files=data_files))
    files.FilesConfig(config, 'fake_package').run()
    self.assertIn(expected, config['files']['data_files'])