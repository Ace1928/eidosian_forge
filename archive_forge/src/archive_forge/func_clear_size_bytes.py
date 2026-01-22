from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def clear_size_bytes(self):
    if self.has_size_bytes_:
        self.has_size_bytes_ = 0
        self.size_bytes_ = 0