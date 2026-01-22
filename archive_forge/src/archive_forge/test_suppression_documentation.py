import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression

        The suppression set on a test method completely overrides a suppression
        with wider scope; if it does not match a warning emitted by that test
        method, the warning is emitted, even if a wider suppression matches.
        