from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_new_value(self):
    if self.has_new_value_:
        self.has_new_value_ = 0
        self.new_value_ = 0