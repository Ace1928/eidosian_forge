import os
import unittest
import doctest
from pygsp.tests import test_graphs, test_filters
from pygsp.tests import test_utils, test_plotting
def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)