import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def check_testsuite_size(self, args, minsize):
    """
        Check that the reported numbers of tests are at least *minsize*.
        """
    lines = self.get_testsuite_listing(args)
    last_line = lines[-1]
    self.assertTrue('tests found' in last_line)
    number = int(last_line.split(' ')[0])
    self.assertIn(len(lines), range(number + 1, number + 20))
    self.assertGreaterEqual(number, minsize)
    return lines