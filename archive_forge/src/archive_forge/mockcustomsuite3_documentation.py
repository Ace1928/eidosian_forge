from twisted.trial import runner, unittest

Mock test module that contains both a C{test_suite} and a C{testSuite} method.
L{runner.TestLoader} should load the tests from the C{testSuite}, not from the
C{Foo} C{TestCase} nor from the C{test_suite} method.

See {twisted.trial.test.test_loader.LoaderTest.test_loadModuleWithBothCustom}.
