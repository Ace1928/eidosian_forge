import struct
from os_ken.lib.pack_utils import msg_pack_into
from . import packet_base
from . import packet_utils
from . import ether_types
@vsid.setter
def vsid(self, vsid):
    self._key = vsid << 8 | self._key & 255
    self._vsid = vsid