from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_timestamp_microseconds_value(self, x):
    self.has_timestamp_microseconds_value_ = 1
    self.timestamp_microseconds_value_ = x