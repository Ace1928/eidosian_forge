import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def getDeprecatedModuleAttribute(self, moduleName, name, version, message=None):
    """
        Retrieve a module attribute which should have been deprecated,
        and assert that we saw the appropriate deprecation warning.

        @type moduleName: C{str}
        @param moduleName: Fully-qualified Python name of the module containing
            the deprecated attribute; if called from the same module as the
            attributes are being deprecated in, using the C{__name__} global can
            be helpful

        @type name: C{str}
        @param name: Attribute name which we expect to be deprecated

        @param version: The first L{version<twisted.python.versions.Version>} that
            the module attribute was deprecated.

        @type message: C{str}
        @param message: (optional) The expected deprecation message for the module attribute

        @return: The given attribute from the named module

        @raise FailTest: if no warnings were emitted on getattr, or if the
            L{DeprecationWarning} emitted did not produce the canonical
            please-use-something-else message that is standard for Twisted
            deprecations according to the given version and replacement.

        @since: Twisted 21.2.0
        """
    fqpn = moduleName + '.' + name
    module = sys.modules[moduleName]
    attr = getattr(module, name)
    warningsShown = self.flushWarnings([self.getDeprecatedModuleAttribute])
    if len(warningsShown) == 0:
        self.fail(f'{fqpn} is not deprecated.')
    observedWarning = warningsShown[0]['message']
    expectedWarning = DEPRECATION_WARNING_FORMAT % {'fqpn': fqpn, 'version': getVersionString(version)}
    if message is not None:
        expectedWarning = expectedWarning + ': ' + message
    self.assert_(observedWarning.startswith(expectedWarning), f'Expected {observedWarning!r} to start with {expectedWarning!r}')
    return attr