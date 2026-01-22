import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def init_unbuffered_query(self):
    """
        :raise OperationalError: If the connection to the MySQL server is lost.
        :raise InternalError:
        """
    self.unbuffered_active = True
    first_packet = self.connection._read_packet()
    if first_packet.is_ok_packet():
        self._read_ok_packet(first_packet)
        self.unbuffered_active = False
        self.connection = None
    elif first_packet.is_load_local_packet():
        self._read_load_local_packet(first_packet)
        self.unbuffered_active = False
        self.connection = None
    else:
        self.field_count = first_packet.read_length_encoded_integer()
        self._get_descriptions()
        self.affected_rows = 18446744073709551615