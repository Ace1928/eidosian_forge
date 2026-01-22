from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def get_wire(self):
    """Return the wire format message."""
    return self.output.getvalue()