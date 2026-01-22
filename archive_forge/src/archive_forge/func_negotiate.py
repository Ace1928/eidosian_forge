from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def negotiate(self, data):
    """
        Got negotiate data.
        """
    command, payload = (data[0:1], data[1:])
    assert isinstance(command, bytes)
    if command == NAWS:
        self.naws(payload)
    else:
        logger.info('Negotiate (%r got bytes)', len(data))