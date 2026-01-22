import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
@contextlib.contextmanager
def testPartExecutor(self, test_case, subTest=False):
    old_success = self.success
    self.success = True
    try:
        yield
    except KeyboardInterrupt:
        raise
    except SkipTest as e:
        self.success = False
        _addSkip(self.result, test_case, str(e))
    except _ShouldStop:
        pass
    except:
        exc_info = sys.exc_info()
        if self.expecting_failure:
            self.expectedFailure = exc_info
        else:
            self.success = False
            if subTest:
                self.result.addSubTest(test_case.test_case, test_case, exc_info)
            else:
                _addError(self.result, test_case, exc_info)
        exc_info = None
    else:
        if subTest and self.success:
            self.result.addSubTest(test_case.test_case, test_case, None)
    finally:
        self.success = self.success and old_success