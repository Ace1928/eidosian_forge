from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase

        When C{D} is cancelled, it won't reach the callbacks added to it by
        application code until C{C} reaches the point in its callback chain
        where C{G} awaits it.  Otherwise, application code won't be able to
        track resource usage that C{D} may be using.
        