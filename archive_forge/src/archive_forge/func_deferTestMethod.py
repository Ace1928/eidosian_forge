import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
def deferTestMethod(self, ignored, result):
    d = self._run(self._testMethodName, result)
    d.addCallbacks(self._cbDeferTestMethod, self._ebDeferTestMethod, callbackArgs=(result,), errbackArgs=(result,))
    d.addBoth(self.deferRunCleanups, result)
    d.addBoth(self.deferTearDown, result)
    return d