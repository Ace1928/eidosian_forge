import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_copyFunction(self) -> None:
    """
        Copying a function returns the same reference, without creating
        an actual copy.
        """
    f = copy.copy(sampleFunction)
    self.assertEqual(f, sampleFunction)