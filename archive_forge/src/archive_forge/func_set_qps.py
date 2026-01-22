from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_qps(self, x):
    self.has_qps_ = 1
    self.qps_ = x