import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
@classmethod
def set_auth_parser(cls, auth_cls):
    cls._auth_parsers[auth_cls.auth_type] = auth_cls