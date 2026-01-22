import time
from twisted.internet import protocol
from twisted.names import dns, resolve
from twisted.python import log
def handleOther(self, message, protocol, address):
    """
        Called by L{DNSServerFactory.messageReceived} when a message with
        unrecognised I{OPCODE} is received.

        Replies with a I{Not Implemented} error by default.

        An error message will be logged if C{DNSServerFactory.verbose} is C{>1}.

        Override in a subclass.

        @param protocol: The DNS protocol instance to which to send a response
            message.
        @type protocol: L{dns.DNSDatagramProtocol} or L{dns.DNSProtocol}

        @param message: The original DNS query message for which a response
            message will be constructed.
        @type message: L{dns.Message}

        @param address: The address to which the response message will be sent
            or L{None} if C{protocol} is a stream protocol.
        @type address: L{tuple} or L{None}
        """
    message.rCode = dns.ENOTIMP
    self.sendReply(protocol, message, address)
    self._verboseLog('Unknown op code (%d) from %r' % (message.opCode, address))