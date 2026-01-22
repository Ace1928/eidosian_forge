from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
def field_setter(self, new_value):
    try:
        new_value = type_checker.CheckValue(new_value)
    except TypeError as e:
        raise TypeError('Cannot set %s to %.1024r: %s' % (field.full_name, new_value, e))
    if not field.has_presence and (not new_value):
        self._fields.pop(field, None)
    else:
        self._fields[field] = new_value
    if not self._cached_byte_size_dirty:
        self._Modified()