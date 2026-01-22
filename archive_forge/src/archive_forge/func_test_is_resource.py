import unittest
import os
import contextlib
import importlib_resources as resources
def test_is_resource(self):
    is_resource = resources.is_resource
    self.assertTrue(is_resource(self.anchor01, 'utf-8.file'))
    self.assertFalse(is_resource(self.anchor01, 'no_such_file'))
    self.assertFalse(is_resource(self.anchor01))
    self.assertFalse(is_resource(self.anchor01, 'subdirectory'))
    for path_parts in self._gen_resourcetxt_path_parts():
        self.assertTrue(is_resource(self.anchor02, *path_parts))