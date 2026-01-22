from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_computed(self, x):
    self.has_computed_ = 1
    self.computed_ = x