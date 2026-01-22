import json
import os
import sys
from io import BytesIO
from twisted.internet import address, error, interfaces, reactor
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, util
from twisted.trial import unittest
from twisted.web import client, http, http_headers, resource, server, twcgi
from twisted.web.http import INTERNAL_SERVER_ERROR, NOT_FOUND
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
import os, sys
import sys
import json
import os
import os
def test_malformedHeaderCGI(self):
    """
        Check for the error message in the duplicated header
        """
    cgiFilename = self.writeCGI(BROKEN_HEADER_CGI)
    portnum = self.startServer(cgiFilename)
    url = 'http://localhost:%d/cgi' % (portnum,)
    url = url.encode('ascii')
    agent = client.Agent(reactor)
    d = agent.request(b'GET', url)
    d.addCallback(discardBody)
    loggedMessages = []

    def addMessage(eventDict):
        loggedMessages.append(log.textFromEventDict(eventDict))
    log.addObserver(addMessage)
    self.addCleanup(log.removeObserver, addMessage)

    def checkResponse(ignored):
        self.assertIn('ignoring malformed CGI header: ' + repr(b'XYZ'), loggedMessages)
    d.addCallback(checkResponse)
    return d