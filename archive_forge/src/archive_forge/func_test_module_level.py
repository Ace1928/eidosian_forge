import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
def test_module_level(self):
    load_tests = test_fixtures.optimize_module_test_loader()
    loader = unittest.TestLoader()
    found_tests = loader.discover(start_dir, pattern='test_fixtures.py')
    new_loader = load_tests(loader, found_tests, 'test_fixtures.py')
    self.assertIsInstance(new_loader, testresources.OptimisingTestSuite)
    actual_tests = unittest.TestSuite(testscenarios.generate_scenarios(found_tests))
    self.assertEqual(new_loader.countTestCases(), actual_tests.countTestCases())