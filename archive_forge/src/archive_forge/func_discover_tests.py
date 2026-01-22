import sys
import unittest
from unittest import TestCase
import faulthandler
from llvmlite.tests import customize
def discover_tests(startdir):
    """Discover test under a directory
    """
    loader = unittest.TestLoader()
    suite = loader.discover(startdir)
    return suite