import sys
from twisted.internet._glibbase import ensureNotImported
from twisted.trial.unittest import TestCase
def test_simulate(self):
    """
        C{simulate} can be called without raising any errors when there are
        no delayed calls for the reactor and hence there is no defined sleep
        period.
        """
    sut = gireactor.PortableGIReactor(useGtk=False)
    self.assertIs(None, sut.timeout())
    sut.simulate()