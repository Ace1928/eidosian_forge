import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
def test_synchronizedName(self):
    """
        The name of a synchronized method is inaffected by the synchronization
        decorator.
        """
    self.assertEqual('aMethod', TestObject.aMethod.__name__)