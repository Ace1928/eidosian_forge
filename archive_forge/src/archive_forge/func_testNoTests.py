import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testNoTests(self):
    self.assertEqual({frozenset(): []}, split_by_resources([]))