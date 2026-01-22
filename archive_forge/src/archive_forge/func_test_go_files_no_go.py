import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_files_no_go(self):
    self.write_file('foo.notgo', 'package main\nfunc main')
    self.assertFalse(self.generate_configs())
    self.assertEqual(os.listdir(self.temp_path), ['foo.notgo'])