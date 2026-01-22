import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase

        L{threadable.isInIOThread} returns C{True} if and only if it is called
        in the same thread as L{threadable.registerAsIOThread}.
        