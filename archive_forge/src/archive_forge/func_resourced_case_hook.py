import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def resourced_case_hook(test):
    self.assertTrue(sample_resource._uses > 0)