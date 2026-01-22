from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def registerAccount(self, username=None, password=None):
    if username:
        self.jid.user = username
    if password:
        self.password = password
    iq = IQ(self.xmlstream, 'set')
    iq.addElement(('jabber:iq:register', 'query'))
    iq.query.addElement('username', content=self.jid.user)
    iq.query.addElement('password', content=self.password)
    iq.addCallback(self._registerResultEvent)
    iq.send()