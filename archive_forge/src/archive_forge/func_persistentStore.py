import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def persistentStore(obj, jel, perst=perst):
    perst[1] = perst[1] + 1
    perst[0][perst[1]] = obj
    return str(perst[1])