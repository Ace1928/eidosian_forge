import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def ext_functest_builder(method, func):
    """Generate a test method that tests the given extension function.

    This is intended to generate test methods that test both a pure-Python
    version and an extension version using common test code. The extension test
    will raise SkipTest if the extension is not found.

    Sample usage:

    class MyTest(TestCase);
        def _do_some_test(self, func_impl):
            self.assertEqual('foo', func_impl())

        test_foo = functest_builder(_do_some_test, foo_py)
        test_foo_extension = ext_functest_builder(_do_some_test, _foo_c)

    Args:
      method: The method to run. It must must two parameters, self and the
        function implementation to test.
      func: The function implementation to pass to method.
    """

    def do_test(self):
        if not isinstance(func, types.BuiltinFunctionType):
            raise SkipTest('%s extension not found' % func)
        method(self, func)
    return do_test