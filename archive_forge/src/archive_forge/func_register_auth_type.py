import hashlib
import operator
import struct
from . import packet_base
from os_ken.lib import stringify
@classmethod
def register_auth_type(cls, auth_type):

    def _set_type(auth_cls):
        auth_cls.set_type(auth_cls, auth_type)
        cls.set_auth_parser(auth_cls)
        return auth_cls
    return _set_type