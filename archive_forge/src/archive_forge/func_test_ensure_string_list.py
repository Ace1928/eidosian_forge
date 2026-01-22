import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def test_ensure_string_list(self):
    cmd = self.cmd
    cmd.not_string_list = ['one', 2, 'three']
    cmd.yes_string_list = ['one', 'two', 'three']
    cmd.not_string_list2 = object()
    cmd.yes_string_list2 = 'ok'
    cmd.ensure_string_list('yes_string_list')
    cmd.ensure_string_list('yes_string_list2')
    self.assertRaises(DistutilsOptionError, cmd.ensure_string_list, 'not_string_list')
    self.assertRaises(DistutilsOptionError, cmd.ensure_string_list, 'not_string_list2')
    cmd.option1 = 'ok,dok'
    cmd.ensure_string_list('option1')
    self.assertEqual(cmd.option1, ['ok', 'dok'])
    cmd.option2 = ['xxx', 'www']
    cmd.ensure_string_list('option2')
    cmd.option3 = ['ok', 2]
    self.assertRaises(DistutilsOptionError, cmd.ensure_string_list, 'option3')