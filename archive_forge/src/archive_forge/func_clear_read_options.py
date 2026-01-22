from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_read_options(self):
    if self.has_read_options_:
        self.has_read_options_ = 0
        if self.read_options_ is not None:
            self.read_options_.Clear()