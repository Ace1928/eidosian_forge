from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_integer_value(self):
    if self.has_integer_value_:
        self.has_integer_value_ = 0
        self.integer_value_ = 0