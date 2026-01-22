from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_items(self, x):
    self.has_items_ = 1
    self.items_ = x