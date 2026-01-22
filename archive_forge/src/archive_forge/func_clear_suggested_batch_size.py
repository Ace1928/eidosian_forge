from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_suggested_batch_size(self):
    if self.has_suggested_batch_size_:
        self.has_suggested_batch_size_ = 0
        self.suggested_batch_size_ = 0