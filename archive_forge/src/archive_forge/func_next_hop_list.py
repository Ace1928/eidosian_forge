import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
@next_hop_list.setter
def next_hop_list(self, addr_list):
    if not isinstance(addr_list, (list, tuple)):
        addr_list = [addr_list]
    for addr in addr_list:
        if not ip.valid_ipv4(addr) and (not ip.valid_ipv6(addr)):
            raise ValueError('Invalid address for next_hop: %s' % addr)
    self._next_hop = addr_list[0]
    self._next_hop_list = addr_list