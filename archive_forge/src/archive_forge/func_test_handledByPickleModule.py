import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_handledByPickleModule(self) -> None:
    """
        Handling L{pickle.PicklingError} handles
        L{_UniversalPicklingError}.
        """
    self.assertRaises(pickle.PicklingError, self.raise_UniversalPicklingError)