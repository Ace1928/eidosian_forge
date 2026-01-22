from typing import Optional
def parseFields(self, name, portAndProtocol, *aliases):
    try:
        port, protocol = portAndProtocol.split('/')
        port = int(port)
    except BaseException:
        raise InvalidServicesConfError(f'Invalid port/protocol: {repr(portAndProtocol)}')
    self.services[name, protocol] = port
    for alias in aliases:
        self.services[alias, protocol] = port