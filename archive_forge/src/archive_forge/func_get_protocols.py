import inspect
import struct
import base64
from . import packet_base
from . import ethernet
from os_ken import utils
from os_ken.lib.stringify import StringifyMixin
def get_protocols(self, protocol):
    """Returns a list of protocols that matches to the specified protocol.
        """
    if isinstance(protocol, packet_base.PacketBase):
        protocol = protocol.__class__
    assert issubclass(protocol, packet_base.PacketBase)
    return [p for p in self.protocols if isinstance(p, protocol)]