from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_set_policy(self):
    if self.has_set_policy_:
        self.has_set_policy_ = 0
        self.set_policy_ = 1