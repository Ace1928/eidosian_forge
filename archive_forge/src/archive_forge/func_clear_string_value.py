from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_string_value(self):
    if self.has_string_value_:
        self.has_string_value_ = 0
        self.string_value_ = ''