from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def mutable_owner(self):
    self.has_owner_ = 1
    return self.owner()