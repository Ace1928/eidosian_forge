from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
from googlecloudsdk.third_party.appengine.datastore.entity_v4_pb import *
import googlecloudsdk.third_party.appengine.datastore.entity_v4_pb
def clear_property_filter(self):
    if self.has_property_filter_:
        self.has_property_filter_ = 0
        if self.property_filter_ is not None:
            self.property_filter_.Clear()