import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@port_description.setter
def port_description(self, value):
    self.tlv_info = value