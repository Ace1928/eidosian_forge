import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def normal_case_hook(test):
    self.assertEqual(sample_resource._uses, 0)