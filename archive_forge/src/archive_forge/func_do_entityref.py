from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_entityref(self, byte):
    if byte.isspace() or byte == '<':
        if self.beExtremelyLenient:
            if self.erefbuf and self.erefbuf != 'amp':
                self.erefextra = self.erefbuf
            self.erefbuf = 'amp'
            if byte == '<':
                return 'tagstart'
            else:
                self.erefextra += byte
                return 'spacebodydata'
        self._parseError('Bad entity reference')
    elif byte != ';':
        self.erefbuf += byte
    else:
        return 'bodydata'