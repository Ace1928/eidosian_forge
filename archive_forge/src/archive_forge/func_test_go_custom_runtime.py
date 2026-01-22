import os
import unittest
import yaml
from gae_ext_runtime import testutil
def test_go_custom_runtime(self):
    self.write_file('foo.go', 'package main\nfunc main')
    self.generate_configs(custom=True)
    self.assert_file_exists_with_contents('app.yaml', 'env: flex\nruntime: go\n')
    self.assert_file_exists_with_contents('Dockerfile', self.read_runtime_def_file('data', 'Dockerfile'))
    self.assert_file_exists_with_contents('.dockerignore', self.read_runtime_def_file('data', 'dockerignore'))