import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@staticmethod
def encode_bridge_id(priority, system_id_extension, mac_address):
    mac_addr = int(binascii.hexlify(addrconv.mac.text_to_bin(mac_address)), 16)
    return (priority + system_id_extension << 48) + mac_addr