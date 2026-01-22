from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
def makeInvalidByDeletingParent(self) -> None:
    """
        Probably not a wise method to call.

        This method removes the :code:`parent` attribute,
        which has to exist in IService classes.
        """
    del self.parent