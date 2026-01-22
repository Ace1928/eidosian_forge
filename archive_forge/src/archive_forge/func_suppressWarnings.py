import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def suppressWarnings(f, *suppressedWarnings):
    """
    Wrap C{f} in a callable which suppresses the indicated warnings before
    invoking C{f} and unsuppresses them afterwards.  If f returns a Deferred,
    warnings will remain suppressed until the Deferred fires.
    """

    @wraps(f)
    def warningSuppressingWrapper(*a, **kw):
        return runWithWarningsSuppressed(suppressedWarnings, f, *a, **kw)
    return warningSuppressingWrapper