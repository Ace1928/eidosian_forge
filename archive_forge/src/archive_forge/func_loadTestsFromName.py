import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def loadTestsFromName(self, name, module=None):
    """Return a suite of all test cases given a string specifier.

        The name may resolve either to a module, a test case class, a
        test method within a test case class, or a callable object which
        returns a TestCase or TestSuite instance.

        The method optionally resolves the names relative to a given module.
        """
    parts = name.split('.')
    error_case, error_message = (None, None)
    if module is None:
        parts_copy = parts[:]
        while parts_copy:
            try:
                module_name = '.'.join(parts_copy)
                module = __import__(module_name)
                break
            except ImportError:
                next_attribute = parts_copy.pop()
                error_case, error_message = _make_failed_import_test(next_attribute, self.suiteClass)
                if not parts_copy:
                    self.errors.append(error_message)
                    return error_case
        parts = parts[1:]
    obj = module
    for part in parts:
        try:
            parent, obj = (obj, getattr(obj, part))
        except AttributeError as e:
            if getattr(obj, '__path__', None) is not None and error_case is not None:
                self.errors.append(error_message)
                return error_case
            else:
                error_case, error_message = _make_failed_test(part, e, self.suiteClass, 'Failed to access attribute:\n%s' % (traceback.format_exc(),))
                self.errors.append(error_message)
                return error_case
    if isinstance(obj, types.ModuleType):
        return self.loadTestsFromModule(obj)
    elif isinstance(obj, type) and issubclass(obj, case.TestCase) and (obj not in (case.TestCase, case.FunctionTestCase)):
        return self.loadTestsFromTestCase(obj)
    elif isinstance(obj, types.FunctionType) and isinstance(parent, type) and issubclass(parent, case.TestCase):
        name = parts[-1]
        inst = parent(name)
        if not isinstance(getattr(inst, name), types.FunctionType):
            return self.suiteClass([inst])
    elif isinstance(obj, suite.TestSuite):
        return obj
    if callable(obj):
        test = obj()
        if isinstance(test, suite.TestSuite):
            return test
        elif isinstance(test, case.TestCase):
            return self.suiteClass([test])
        else:
            raise TypeError('calling %s returned %s, not a test' % (obj, test))
    else:
        raise TypeError("don't know how to make test from: %s" % obj)