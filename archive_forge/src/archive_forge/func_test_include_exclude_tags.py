import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def test_include_exclude_tags(self):

    def get_count(arg_list):
        lines = self.get_testsuite_listing(arg_list)
        self.assertIn('tests found', lines[-1])
        count = int(lines[-1].split()[0])
        self.assertTrue(count > 0)
        return count
    tags = ['long_running', 'long_running, important']
    total = get_count(['numba.tests'])
    for tag in tags:
        included = get_count(['--tags', tag, 'numba.tests'])
        excluded = get_count(['--exclude-tags', tag, 'numba.tests'])
        self.assertEqual(total, included + excluded)
        included = get_count(['--tags=%s' % tag, 'numba.tests'])
        excluded = get_count(['--exclude-tags=%s' % tag, 'numba.tests'])
        self.assertEqual(total, included + excluded)