from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_federated_identity(self):
    if self.has_federated_identity_:
        self.has_federated_identity_ = 0
        self.federated_identity_ = ''