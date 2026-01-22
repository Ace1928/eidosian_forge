import struct
from io import BytesIO
from paramiko import util
from paramiko.common import zero_byte, max_byte, one_byte
from paramiko.util import u
def get_remainder(self):
    """
        Return the `bytes` of this message that haven't already been parsed and
        returned.
        """
    position = self.packet.tell()
    remainder = self.packet.read()
    self.packet.seek(position)
    return remainder