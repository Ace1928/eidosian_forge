import io
import sys
import unittest
def test_setup_teardown_order_with_pathological_suite(self):
    results = []

    class Module1(object):

        @staticmethod
        def setUpModule():
            results.append('Module1.setUpModule')

        @staticmethod
        def tearDownModule():
            results.append('Module1.tearDownModule')

    class Module2(object):

        @staticmethod
        def setUpModule():
            results.append('Module2.setUpModule')

        @staticmethod
        def tearDownModule():
            results.append('Module2.tearDownModule')

    class Test1(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            results.append('setup 1')

        @classmethod
        def tearDownClass(cls):
            results.append('teardown 1')

        def testOne(self):
            results.append('Test1.testOne')

        def testTwo(self):
            results.append('Test1.testTwo')

    class Test2(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            results.append('setup 2')

        @classmethod
        def tearDownClass(cls):
            results.append('teardown 2')

        def testOne(self):
            results.append('Test2.testOne')

        def testTwo(self):
            results.append('Test2.testTwo')

    class Test3(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            results.append('setup 3')

        @classmethod
        def tearDownClass(cls):
            results.append('teardown 3')

        def testOne(self):
            results.append('Test3.testOne')

        def testTwo(self):
            results.append('Test3.testTwo')
    Test1.__module__ = Test2.__module__ = 'Module'
    Test3.__module__ = 'Module2'
    sys.modules['Module'] = Module1
    sys.modules['Module2'] = Module2
    first = unittest.TestSuite((Test1('testOne'),))
    second = unittest.TestSuite((Test1('testTwo'),))
    third = unittest.TestSuite((Test2('testOne'),))
    fourth = unittest.TestSuite((Test2('testTwo'),))
    fifth = unittest.TestSuite((Test3('testOne'),))
    sixth = unittest.TestSuite((Test3('testTwo'),))
    suite = unittest.TestSuite((first, second, third, fourth, fifth, sixth))
    runner = self.getRunner()
    result = runner.run(suite)
    self.assertEqual(result.testsRun, 6)
    self.assertEqual(len(result.errors), 0)
    self.assertEqual(results, ['Module1.setUpModule', 'setup 1', 'Test1.testOne', 'Test1.testTwo', 'teardown 1', 'setup 2', 'Test2.testOne', 'Test2.testTwo', 'teardown 2', 'Module1.tearDownModule', 'Module2.setUpModule', 'setup 3', 'Test3.testOne', 'Test3.testTwo', 'teardown 3', 'Module2.tearDownModule'])