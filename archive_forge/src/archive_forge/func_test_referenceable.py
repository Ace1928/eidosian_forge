import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_referenceable(self):
    """
        A L{pb.Referenceable} instance jellies to a structure which unjellies to
        a L{pb.RemoteReference}.  The C{RemoteReference} has a I{luid} that
        matches up with the local object key in the L{pb.Broker} which sent the
        L{Referenceable}.
        """
    ref = pb.Referenceable()
    jellyBroker = pb.Broker()
    jellyBroker.makeConnection(StringTransport())
    j = jelly.jelly(ref, invoker=jellyBroker)
    unjellyBroker = pb.Broker()
    unjellyBroker.makeConnection(StringTransport())
    uj = jelly.unjelly(j, invoker=unjellyBroker)
    self.assertIn(uj.luid, jellyBroker.localObjects)