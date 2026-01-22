from twisted.protocols import basic
def forwardQuery(self, slash_w, user, host):
    self._refuseMessage(b'Finger forwarding service denied')