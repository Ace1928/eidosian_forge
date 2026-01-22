from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish

        Test basic operations of exceptionFromStreamError.

        Given a realistic stream error, check if a sane exception is returned.

        Using this error::

          <stream:error xmlns:stream='http://etherx.jabber.org/streams'>
            <xml-not-well-formed xmlns='urn:ietf:params:xml:ns:xmpp-streams'/>
          </stream:error>
        