import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def verifyConstructorArgument(testCase, cls, argName, defaultVal, altVal, attrName=None):
    """
    Verify that an attribute has the expected default value and that a
    corresponding argument passed to a constructor is assigned to that
    attribute.

    @param testCase: The L{TestCase} whose assert methods will be
        called.
    @type testCase: L{unittest.TestCase}

    @param cls: The constructor under test.
    @type cls: L{type}

    @param argName: The name of the constructor argument under test.
    @type argName: L{str}

    @param defaultVal: The expected default value of C{attrName} /
        C{argName}
    @type defaultVal: L{object}

    @param altVal: A value which is different from the default. Used to
        test that supplied constructor arguments are actually assigned to the
        correct attribute.
    @type altVal: L{object}

    @param attrName: The name of the attribute under test if different
        from C{argName}. Defaults to C{argName}
    @type attrName: L{str}
    """
    if attrName is None:
        attrName = argName
    actual = {}
    expected = {'defaultVal': defaultVal, 'altVal': altVal}
    o = cls()
    actual['defaultVal'] = getattr(o, attrName)
    o = cls(**{argName: altVal})
    actual['altVal'] = getattr(o, attrName)
    testCase.assertEqual(expected, actual)