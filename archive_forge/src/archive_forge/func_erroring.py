from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
@inlineCallbacks
def erroring():
    yield 'forcing generator'
    raise Exception('Error Marker')