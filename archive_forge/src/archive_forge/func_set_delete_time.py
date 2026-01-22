from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_delete_time(self, x):
    self.has_delete_time_ = 1
    self.delete_time_ = x