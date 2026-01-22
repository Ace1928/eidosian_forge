from zope.interface import implementer
from twisted.pair import ethernet, raw
from twisted.python import components
from twisted.trial import unittest
Adding a protocol with a number >=2**16 raises an exception.