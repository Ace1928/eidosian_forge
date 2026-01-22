from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_boolean_value(self):
    if self.has_boolean_value_:
        self.has_boolean_value_ = 0
        self.boolean_value_ = 0