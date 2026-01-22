from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_expectcdata(self, byte):
    self.cdatabuf += byte
    cdb = self.cdatabuf
    cd = '[CDATA['
    if len(cd) > len(cdb):
        if cd.startswith(cdb):
            return
        elif self.beExtremelyLenient:
            return 'waitforgt'
        else:
            self._parseError('Mal-formed CDATA header')
    if cd == cdb:
        self.cdatabuf = ''
        return 'cdata'
    self._parseError('Mal-formed CDATA header')