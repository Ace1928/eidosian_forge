from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_federated_provider(self):
    if self.has_federated_provider_:
        self.has_federated_provider_ = 0
        self.federated_provider_ = ''