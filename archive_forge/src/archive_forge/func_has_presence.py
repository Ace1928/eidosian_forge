import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
@property
def has_presence(self):
    """Whether the field distinguishes between unpopulated and default values.

    Raises:
      RuntimeError: singular field that is not linked with message nor file.
    """
    if self.label == FieldDescriptor.LABEL_REPEATED:
        return False
    if self.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE or self.containing_oneof:
        return True
    return self._GetFeatures().field_presence != _FEATURESET_FIELD_PRESENCE_IMPLICIT