import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
@property
def fields_by_camelcase_name(self):
    """Same FieldDescriptor objects as in :attr:`fields`, but indexed by
    :attr:`FieldDescriptor.camelcase_name`.
    """
    if self._fields_by_camelcase_name is None:
        self._fields_by_camelcase_name = dict(((f.camelcase_name, f) for f in self.fields))
    return self._fields_by_camelcase_name