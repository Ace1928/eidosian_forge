from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_items(self):
    if self.has_items_:
        self.has_items_ = 0
        self.items_ = 0