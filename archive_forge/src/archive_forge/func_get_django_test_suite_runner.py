from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def get_django_test_suite_runner():
    global DJANGO_TEST_SUITE_RUNNER
    if DJANGO_TEST_SUITE_RUNNER:
        return DJANGO_TEST_SUITE_RUNNER
    try:
        import django
        from django.test.runner import DiscoverRunner

        class MyDjangoTestSuiteRunner(DiscoverRunner):

            def __init__(self, on_run_suite):
                django.setup()
                DiscoverRunner.__init__(self)
                self.on_run_suite = on_run_suite

            def build_suite(self, *args, **kwargs):
                pass

            def suite_result(self, *args, **kwargs):
                pass

            def run_suite(self, *args, **kwargs):
                self.on_run_suite()
    except:
        try:
            from django.test.simple import DjangoTestSuiteRunner
        except:

            class DjangoTestSuiteRunner:

                def __init__(self):
                    pass

                def run_tests(self, *args, **kwargs):
                    raise AssertionError("Unable to run suite with django.test.runner.DiscoverRunner nor django.test.simple.DjangoTestSuiteRunner because it couldn't be imported.")

        class MyDjangoTestSuiteRunner(DjangoTestSuiteRunner):

            def __init__(self, on_run_suite):
                DjangoTestSuiteRunner.__init__(self)
                self.on_run_suite = on_run_suite

            def build_suite(self, *args, **kwargs):
                pass

            def suite_result(self, *args, **kwargs):
                pass

            def run_suite(self, *args, **kwargs):
                self.on_run_suite()
    DJANGO_TEST_SUITE_RUNNER = MyDjangoTestSuiteRunner
    return DJANGO_TEST_SUITE_RUNNER