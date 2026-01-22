import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_lambdaRaisesPicklingError(self) -> None:
    """
        Pickling a C{lambda} function ought to raise a L{pickle.PicklingError}.
        """
    self.assertRaises(pickle.PicklingError, pickle.dumps, lambdaExample)