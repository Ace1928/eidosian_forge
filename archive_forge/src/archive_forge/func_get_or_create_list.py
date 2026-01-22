import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def get_or_create_list(self, key):
    """Returns a list for this key, creating if it didn't exist already."""
    if not self.fields[key].HasField('list_value'):
        self.fields[key].list_value.Clear()
    return self.fields[key].list_value