import unittest
import pulp
from pulp.tests import test_pulp, test_examples, test_gurobipy_env
def get_test_suite(test_docs=False):
    loader = unittest.TestLoader()
    suite_all = unittest.TestSuite()
    pulp_solver_tests = loader.loadTestsFromModule(test_pulp)
    suite_all.addTests(pulp_solver_tests)
    gurobipy_env = loader.loadTestsFromModule(test_gurobipy_env)
    suite_all.addTests(gurobipy_env)
    if test_docs:
        docs_examples = loader.loadTestsFromTestCase(test_examples.Examples_DocsTests)
        suite_all.addTests(docs_examples)
    return suite_all