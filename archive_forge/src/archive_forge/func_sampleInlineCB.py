from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
@inlineCallbacks
def sampleInlineCB(self, getChildDeferred=None):
    """
        Generator for testing cascade cancelling cases.

        @param getChildDeferred: Some callable returning L{Deferred} that we
            awaiting (with C{yield})
        """
    if getChildDeferred is None:
        getChildDeferred = self.getDeferred
    try:
        x = (yield getChildDeferred())
    except UntranslatedError:
        raise TranslatedError()
    except DontFail as df:
        x = df.actualValue - 2
    returnValue(x + 1)