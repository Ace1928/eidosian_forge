from twisted.application import internet
from twisted.trial import unittest
from twisted.words import xmpproutertap as tap
from twisted.words.protocols.jabber import component
def test_makeService(self) -> None:
    """
        The service gets set up with a router and factory.
        """
    opt = tap.Options()
    opt.parseOptions([])
    s = tap.makeService(opt)
    self.assertIsInstance(s, internet.StreamServerEndpointService)
    self.assertEqual('127.0.0.1', s.endpoint._interface)
    self.assertEqual(5347, s.endpoint._port)
    factory = s.factory
    self.assertIsInstance(factory, component.XMPPComponentServerFactory)
    self.assertIsInstance(factory.router, component.Router)
    self.assertEqual('secret', factory.secret)
    self.assertFalse(factory.logTraffic)