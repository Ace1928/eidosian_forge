from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_longitude(self):
    if self.has_longitude_:
        self.has_longitude_ = 0
        self.longitude_ = 0.0