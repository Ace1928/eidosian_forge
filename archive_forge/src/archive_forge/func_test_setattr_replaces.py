import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_setattr_replaces(self):
    """ScopeReplacer can create an instance in local scope.

        An object should appear in globals() by constructing a ScopeReplacer,
        and it will be replaced with the real object upon the first request.
        """
    actions = []
    TestClass.use_actions(actions)

    def factory(replacer, scope, name):
        return TestClass()
    try:
        test_obj6
    except NameError:
        pass
    else:
        self.fail('test_obj6 was not supposed to exist yet')
    lazy_import.ScopeReplacer(scope=globals(), name='test_obj6', factory=factory)
    self.assertEqual(lazy_import.ScopeReplacer, object.__getattribute__(test_obj6, '__class__'))
    test_obj6.bar = 'test'
    self.assertNotEqual(lazy_import.ScopeReplacer, object.__getattribute__(test_obj6, '__class__'))
    self.assertEqual('test', test_obj6.bar)