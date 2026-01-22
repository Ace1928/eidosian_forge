from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def set_federated_identity(self, x):
    self.has_federated_identity_ = 1
    self.federated_identity_ = x