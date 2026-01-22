from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_timestamp_microseconds_value(self):
    if self.has_timestamp_microseconds_value_:
        self.has_timestamp_microseconds_value_ = 0
        self.timestamp_microseconds_value_ = 0