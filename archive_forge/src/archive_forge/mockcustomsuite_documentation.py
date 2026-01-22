from twisted.trial import runner, unittest

Mock test module that contains a C{test_suite} method. L{runner.TestLoader}
should load the tests from the C{test_suite}, not from the C{Foo} C{TestCase}.

See {twisted.trial.test.test_loader.LoaderTest.test_loadModuleWith_test_suite}.
