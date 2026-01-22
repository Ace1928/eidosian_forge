import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def check_listing_prefix(self, prefix):
    listing = self.get_testsuite_listing([prefix])
    for ln in listing[:-1]:
        errmsg = '{!r} not startswith {!r}'.format(ln, prefix)
        self.assertTrue(ln.startswith(prefix), msg=errmsg)