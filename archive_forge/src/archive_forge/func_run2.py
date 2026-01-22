import types
from twisted.internet.defer import Deferred, ensureDeferred, fail, succeed
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def run2():
    d = succeed('foo')
    res = (yield from d)
    return res