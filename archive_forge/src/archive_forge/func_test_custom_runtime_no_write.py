import os
import unittest
from gae_ext_runtime import testutil
def test_custom_runtime_no_write(self):
    """Ensure custom runtime writes app.yaml to disk with GenerateConfigData."""
    self.write_file('Dockerfile', 'boring contents')
    self.generate_config_data()
    self.assert_file_exists_with_contents('app.yaml', 'env: flex\nruntime: custom\n')