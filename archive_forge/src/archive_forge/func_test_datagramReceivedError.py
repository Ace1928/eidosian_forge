import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_datagramReceivedError(self):
    """
        When datagramReceived raises an exception it is logged but the port
        is not disconnected.
        """
    finalDeferred = defer.Deferred()

    def cbCompleted(ign):
        """
            Flush the exceptions which the reactor should have logged and make
            sure they're actually there.
            """
        errs = self.flushLoggedErrors(BadClientError)
        self.assertEqual(len(errs), 2, 'Incorrectly found %d errors, expected 2' % (len(errs),))
    finalDeferred.addCallback(cbCompleted)
    client = BadClient()
    port = reactor.listenUDP(0, client, interface='127.0.0.1')

    def cbCleanup(result):
        """
            Disconnect the port we started and pass on whatever was given to us
            in case it was a Failure.
            """
        return defer.maybeDeferred(port.stopListening).addBoth(lambda ign: result)
    finalDeferred.addBoth(cbCleanup)
    addr = port.getHost()
    attempts = list(range(60))
    succeededAttempts = []

    def makeAttempt():
        """
            Send one packet to the listening BadClient.  Set up a 0.1 second
            timeout to do re-transmits in case the packet is dropped.  When two
            packets have been received by the BadClient, stop sending and let
            the finalDeferred's callbacks do some assertions.
            """
        if not attempts:
            try:
                self.fail('Not enough packets received')
            except Exception:
                finalDeferred.errback()
        self.failIfIdentical(client.transport, None, 'UDP Protocol lost its transport')
        packet = b'%d' % (attempts.pop(0),)
        packetDeferred = defer.Deferred()
        client.setDeferred(packetDeferred)
        client.transport.write(packet, (addr.host, addr.port))

        def cbPacketReceived(packet):
            """
                A packet arrived.  Cancel the timeout for it, record it, and
                maybe finish the test.
                """
            timeoutCall.cancel()
            succeededAttempts.append(packet)
            if len(succeededAttempts) == 2:
                reactor.callLater(0, finalDeferred.callback, None)
            else:
                makeAttempt()

        def ebPacketTimeout(err):
            """
                The packet wasn't received quickly enough.  Try sending another
                one.  It doesn't matter if the packet for which this was the
                timeout eventually arrives: makeAttempt throws away the
                Deferred on which this function is the errback, so when
                datagramReceived callbacks, so it won't be on this Deferred, so
                it won't raise an AlreadyCalledError.
                """
            makeAttempt()
        packetDeferred.addCallbacks(cbPacketReceived, ebPacketTimeout)
        packetDeferred.addErrback(finalDeferred.errback)
        timeoutCall = reactor.callLater(0.1, packetDeferred.errback, error.TimeoutError('Timed out in testDatagramReceivedError'))
    makeAttempt()
    return finalDeferred