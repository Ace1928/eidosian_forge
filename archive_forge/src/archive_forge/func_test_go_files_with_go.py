import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_files_with_go(self):
    self.write_file('foo.go', 'package main\nfunc main')
    self.generate_configs()
    with open(self.full_path('app.yaml')) as f:
        contents = yaml.load(f)
    self.assertEqual(contents, {'runtime': 'go', 'env': 'flex'})
    self.assert_no_file('Dockerfile')
    self.assert_no_file('.dockerignore')
    self.generate_configs(deploy=True)
    self.assert_file_exists_with_contents('Dockerfile', self.read_runtime_def_file('data', 'Dockerfile'))
    self.assert_file_exists_with_contents('.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))