import os
import unittest
from gae_ext_runtime import testutil
def test_generate_without_ruby_files_no_write(self):
    """Tests generate_config_data does nothing if no ruby files."""
    self.write_file('index.html', 'index')
    self.assertIsNone(self.generate_config_data())
    self.assertFalse(os.path.exists(self.full_path('app.yaml')))