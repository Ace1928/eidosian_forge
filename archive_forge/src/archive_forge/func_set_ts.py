from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_ts(self, x):
    self.has_ts_ = 1
    self.ts_ = x