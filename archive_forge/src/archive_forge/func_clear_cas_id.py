from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_cas_id(self):
    if self.has_cas_id_:
        self.has_cas_id_ = 0
        self.cas_id_ = 0