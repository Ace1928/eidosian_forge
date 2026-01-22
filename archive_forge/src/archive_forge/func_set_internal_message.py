from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_internal_message(self, x):
    self.has_internal_message_ = 1
    self.internal_message_ = x