import ipaddress
import struct
from ipaddress import IPv4Address, IPv6Address
from os import PathLike
from typing import Any, AnyStr, Dict, IO, List, Optional, Tuple, Union
from maxminddb.const import MODE_AUTO, MODE_MMAP, MODE_FILE, MODE_MEMORY, MODE_FD
from maxminddb.decoder import Decoder
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def get_with_prefix_len(self, ip_address: Union[str, IPv6Address, IPv4Address]) -> Tuple[Optional[Record], int]:
    """Return a tuple with the record and the associated prefix length


        Arguments:
        ip_address -- an IP address in the standard string notation
        """
    if isinstance(ip_address, str):
        address = ipaddress.ip_address(ip_address)
    else:
        address = ip_address
    try:
        packed_address = bytearray(address.packed)
    except AttributeError as ex:
        raise TypeError('argument 1 must be a string or ipaddress object') from ex
    if address.version == 6 and self._metadata.ip_version == 4:
        raise ValueError(f'Error looking up {ip_address}. You attempted to look up an IPv6 address in an IPv4-only database.')
    pointer, prefix_len = self._find_address_in_tree(packed_address)
    if pointer:
        return (self._resolve_data_pointer(pointer), prefix_len)
    return (None, prefix_len)