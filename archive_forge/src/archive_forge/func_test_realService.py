from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
def test_realService(self) -> None:
    """
        Service implements IService.
        """
    myService = Service()
    verifyObject(IService, myService)