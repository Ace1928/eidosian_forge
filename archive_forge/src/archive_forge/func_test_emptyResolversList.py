from twisted.names.error import DomainError
from twisted.names.resolve import ResolverChain
from twisted.trial.unittest import TestCase
def test_emptyResolversList(self) -> None:
    """
        L{ResolverChain._lookup} returns a L{DomainError} failure if
        its C{resolvers} list is empty.
        """
    r = ResolverChain([])
    d = r.lookupAddress('www.example.com')
    f = self.failureResultOf(d)
    self.assertIs(f.trap(DomainError), DomainError)