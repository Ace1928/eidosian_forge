from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport

        Test that it can fail sensibly when someone tried to connect before
        we did.
        