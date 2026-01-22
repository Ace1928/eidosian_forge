from twisted.names.error import DomainError
from twisted.names.resolve import ResolverChain
from twisted.trial.unittest import TestCase
def test_emptyResolversListLookupAllRecords(self) -> None:
    """
        L{ResolverChain.lookupAllRecords} returns a L{DomainError}
        failure if its C{resolvers} list is empty.
        """
    r = ResolverChain([])
    d = r.lookupAllRecords('www.example.com')
    f = self.failureResultOf(d)
    self.assertIs(f.trap(DomainError), DomainError)