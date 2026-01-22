import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def get_or_create_struct(self, key):
    """Returns a struct for this key, creating if it didn't exist already."""
    if not self.fields[key].HasField('struct_value'):
        self.fields[key].struct_value.Clear()
    return self.fields[key].struct_value