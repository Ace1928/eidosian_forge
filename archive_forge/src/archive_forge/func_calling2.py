from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
@inlineCallbacks
def calling2():
    yield calling3()